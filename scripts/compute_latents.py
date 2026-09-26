"""
Compute latents for a dataset and, in the same pass, compute channel-wise statistics.
All latents and the statistics are saved, then packaged into a single tarball.
"""

import argparse
import os
import tarfile
import numpy as np
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import ml_collections

from src.latent import compute_latent_representation, get_latent_model
from src.data import (
    get_distributed_image_dataloader,
    get_data,
)


def setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def process(
    rank: int,
    world_size: int,
    file_list: List[str],
    model: torch.nn.Module,
    config,
    save_path: str,
    stats_name: str,
) -> None:
    """Worker that computes latents, saves them, and accumulates channel-wise stats.

    Accumulates per-channel sums and squared sums across all pixels to compute mean/std.
    """
    setup(rank, world_size, config.master_port)

    dataloader = get_distributed_image_dataloader(
        file_list,
        rank,
        world_size,
        config,
        base_name=getattr(config, "basedir", None),
    )
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model.eval()

    # Running stats on this rank
    running_sum = None  # float64 [C]
    running_sum_sq = None  # float64 [C]
    running_count = torch.zeros(1, dtype=torch.float64, device=device)

    iterator = tqdm(
        dataloader,
        desc=f"Rank {rank}/{world_size} computing latents",
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
        disable=(rank != 0),
    )

    with torch.no_grad():
        for images, _, paths in iterator:
            images = images.to(device, non_blocking=True)
            # Ensure images match model dtype (e.g. float16 if using Flux on GPU)
            images = images.to(next(model.parameters()).dtype)
            image_latents = compute_latent_representation(
                images, model, config.batch_size
            )  # [B, C, H, W]

            # Accumulate per-channel stats on latents
            latents = image_latents.to(torch.float32)
            batch_sum = latents.sum(dim=(0, 2, 3)).to(torch.float64)  # [C]
            batch_sum_sq = (latents.square()).sum(dim=(0, 2, 3)).to(torch.float64)  # [C]
            b, _, h, w = latents.shape
            batch_count = torch.tensor([float(b * h * w)], dtype=torch.float64, device=device)

            if running_sum is None:
                running_sum = torch.zeros_like(batch_sum, dtype=torch.float64, device=device)
                running_sum_sq = torch.zeros_like(batch_sum_sq, dtype=torch.float64, device=device)

            running_sum += batch_sum
            running_sum_sq += batch_sum_sq
            running_count += batch_count

            # Save each latent for every image in the batch
            for i, rel_path in enumerate(paths):
                img_save_path = os.path.join(save_path, rel_path + ".pt")
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                torch.save(image_latents[i].cpu(), img_save_path)

    # Distributed reduction to get global sums across ranks
    if running_sum is None:
        # No data on this rank; create zero tensors for reduction
        running_sum = torch.zeros(1, dtype=torch.float64, device=device)
        running_sum_sq = torch.zeros(1, dtype=torch.float64, device=device)

    # Align shapes across ranks for all_reduce
    # Broadcast channel dimension size from rank 0 if needed
    num_channels = torch.tensor([running_sum.numel()], dtype=torch.int64, device=device)
    dist.broadcast(num_channels, src=0)
    if running_sum.numel() != int(num_channels.item()):
        running_sum = torch.zeros(int(num_channels.item()), dtype=torch.float64, device=device)
        running_sum_sq = torch.zeros(int(num_channels.item()), dtype=torch.float64, device=device)

    dist.all_reduce(running_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_sq, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_count, op=dist.ReduceOp.SUM)

    # Only rank 0 computes and saves the statistics
    if rank == 0:
        eps = 1e-12
        mean64 = running_sum / running_count  # [C] float64 on device
        var64 = (running_sum_sq / running_count) - mean64.square()
        var64 = torch.clamp(var64, min=eps)
        mean = mean64.to(torch.float32).cpu()
        std = torch.sqrt(var64).to(torch.float32).cpu()

        torch.save(mean, os.path.join(save_path, f"{stats_name}_channel_mean.pt"))
        torch.save(std, os.path.join(save_path, f"{stats_name}_channel_std.pt"))
        print(f"Saved channel-wise mean/std next to latents for packaging: {os.path.join(save_path, f'{stats_name}_channel_mean.pt')}")

    cleanup()


def apply_norm(
    rank: int,
    world_size: int, 
    file_list,  # may be List[str] (expected) or a CSV path; we handle List[str]
    config,
    save_path: str,
    stats_name: str,
): 
    setup(rank, world_size, config.master_port)

    # Load stats computed in `process`
    mean = torch.load(os.path.join(save_path, f"{stats_name}_channel_mean.pt"))  # [C]
    std = torch.load(os.path.join(save_path, f"{stats_name}_channel_std.pt"))    # [C]
    TARGET_STD = 0.5

    # Prepare affine params as float32 CPU tensors for broadcasting
    mean = mean.to(torch.float32).view(1, -1, 1, 1)
    std = std.to(torch.float32).clamp(min=1e-12)
    scale = (torch.tensor(TARGET_STD, dtype=torch.float32) / std).view(1, -1, 1, 1)  # [1,C,1,1]
    bias = (torch.zeros_like(mean) - mean * scale) # [1,C,1,1]

    # Robustly ensure we have a Python list of relative paths
    if isinstance(file_list, str):
        # If a CSV path ever slipped through, fall back to loader
        paths_only, _ = get_data(config)
    else:
        paths_only = file_list

    # Deterministic split across ranks
    total = len(paths_only)
    indices = list(range(total))
    shard = indices[rank::world_size]


    # Rank 0 progress bar
    iterator = shard
    if rank == 0:
        iterator = tqdm(shard, desc=f"Rank {rank}/{world_size} applying norm", bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}")

    def apply_affine(x: torch.Tensor) -> torch.Tensor:
        """
        x: [C,H,W] or [1,C,H,W] float32/float16/float64
        returns x' with mean 0 and std TARGET_STD per channel
        """
        x_b = x.unsqueeze(0).to(torch.float32)
        x_b = x_b * scale + bias
        return x_b.squeeze(0)

    # Process files for this rank
    skipped = 0
    for idx in iterator:
        rel_path = paths_only[idx]
        latent_path = os.path.join(save_path, rel_path + ".pt")
        if not os.path.exists(latent_path):
            skipped += 1
            continue
        lat = torch.load(latent_path, map_location="cpu")  # expected [C,H,W]
        lat = apply_affine(lat)
        # Overwrite in place
        os.makedirs(os.path.dirname(latent_path), exist_ok=True)
        torch.save(lat, latent_path)

    if rank == 0:
        print(f"Normalization done on rank 0.")

    cleanup()


def create_tarball_from_directory(source_dir: str, tarball_path: str) -> None:
    with tarfile.open(tarball_path, "w:gz") as tar:
        for root, _, files in os.walk(source_dir):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, start=source_dir)
                tar.add(full_path, arcname=arcname)


def run(config) -> None:
    world_size = torch.cuda.device_count()

    # Only allow CSV input
    if not config.filelist.endswith(".csv"):
        raise ValueError("Only CSV files are supported as input. Please provide a .csv file.")
    
    file_list, _ = get_data(config)

    # Output directory for raw .pt latents and stats (to be tarred after)
    latents_output_dir = getattr(config, "output_latents", None)
    if latents_output_dir is None:
        latents_output_dir = os.path.join(os.path.dirname(config.filelist), "Latents")
    os.makedirs(latents_output_dir, exist_ok=True)
    print(f"Saving latents to {latents_output_dir}")

    # Model
    ckpt_path = getattr(config, "ckpt_path", None)
    model = get_latent_model(path=config.vae_path, modality=config.modality, ckpt_path=ckpt_path)

    # Determine stats paths
    mean_save_path = os.path.join(latents_output_dir, f"{config.stats_name}_channel_mean.pt")
    std_save_path = os.path.join(latents_output_dir, f"{config.stats_name}_channel_std.pt")

    # SKIP LOGIC: check for last image and stats
    last_latent_path = os.path.join(latents_output_dir, file_list[-1] + ".pt")
    if os.path.exists(mean_save_path) and os.path.exists(std_save_path) and os.path.exists(last_latent_path):
        print(f"Skipping compute_latents because last image latent and stats already exist.")
        # We still need to create tarball if it's missing or handle the cleanup if requested
    else:
        # Spawn workers
        mp.spawn(
            process,
            args=(
                world_size,
                file_list,
                model,
                config,
                latents_output_dir,
                config.stats_name,
            ),
            nprocs=world_size,
            join=True,
        )

        # Spawn workers for normalization unless skipped
        if not config.skip_norm:
            mp.spawn(
                apply_norm,
                args=(
                    world_size,
                    file_list,
                    config,
                    latents_output_dir,
                    config.stats_name,
                ),
                nprocs=world_size,
                join=True,
            )
        else:
            print("Skipping normalization as requested.")

    # Package into a tarball
    tarball_path = None
    if not config.skip_tarball:
        print("Creating tarball")
        tarball_name = getattr(config, "tarball_name", None)
        if tarball_name is None:
            base = os.path.basename(os.path.normpath(latents_output_dir))
            tarball_name = base + ".tar.gz"
        tarball_path = os.path.join(os.path.dirname(latents_output_dir), tarball_name)
        create_tarball_from_directory(latents_output_dir, tarball_path)
        print(f"Created tarball: {tarball_path}")

    # Load and output stats for convenience
    mean_path = os.path.join(latents_output_dir, f"{config.stats_name}_channel_mean.pt")
    std_path = os.path.join(latents_output_dir, f"{config.stats_name}_channel_std.pt")
    if os.path.exists(mean_path):
        mean = torch.load(mean_path, weights_only=True)
        print(f"Channel-wise mean: {mean}")
    if os.path.exists(std_path):
        std = torch.load(std_path, weights_only=True)
        print(f"Channel-wise std: {std}")


    # Always remove the directory after creating tarball - keep only the tarball
    # But if skip_tarball is True, we must NOT remove the directory, regardless of remove_files flag.
    if config.remove_files and not config.skip_tarball: 
        import shutil
        shutil.rmtree(latents_output_dir)
        print(f"Removed directory {latents_output_dir} after packaging. Only tarball remains: {tarball_path}")
    elif config.remove_files and config.skip_tarball:
        print("Warning: remove_files was set to True but skip_tarball is also True. Keeping files to avoid data loss.")

    # Post-run instructions: how to extract the tarball
    if tarball_path:
        print(
            "\nTo extract into a new folder:\n"
            f"  mkdir -p output/latents \n"
            f"  tar -xzf {tarball_path} -C output/latents\n\n"
            "To list contents without extracting:\n"
            f"  tar -tzf {tarball_path}\n\n"
            "To extract only the stats files:\n"
            f"  mkdir -p output/latents\n"
            f"  tar -xzf {tarball_path} -C outputs/latents channel_mean.pt channel_std.pt\n"
        )
    else:
         print(f"Latents are ready in {latents_output_dir}")


def get_args():
    parser = argparse.ArgumentParser(description="Compute latents and channel-wise stats, then tar them.")
    parser.add_argument("--basedir", type=str, required=True, help="Base directory used to resolve relative paths in the filelist CSV")
    parser.add_argument("--output_latents", type=str, help="Directory where the tarball will be saved. Also tmpdir for single files so high filecount necessary!")
    parser.add_argument("--filelist", type=str, help="CSV file with paths to data")
    parser.add_argument("--tarball_name", type=str, default=None, help="Name for the output tar.gz (default: <output_dir>.tar.gz)")
    parser.add_argument("--vae_path", type=str, default="black-forest-labs/FLUX.2-dev", help="Path or repo ID for the VAE model")
    parser.add_argument("--modality", type=str, default="xray", help="Modality for MedVAE (xray, ct, mri)")
    parser.add_argument("--stats_name", type=str, default="latents", help="Name for the output stats (default: latents_{mean/std}.pt)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--remove_files", action="store_true", help="Remove generated latents after compuating stats (Tarball remains).")
    parser.add_argument("--skip_tarball", action="store_true", help="Skip creating a tarball of the latents.")
    parser.add_argument("--skip_norm", action="store_true", help="Compute stats but do not apply normalization to the latents.")
    parser.add_argument("--master_port", type=int, default=12344)
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to finetuned model checkpoint (model.pt from medvae_finetune)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = ml_collections.ConfigDict(vars(args)) 
    run(config)


