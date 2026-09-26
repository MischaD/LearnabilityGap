import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import ml_collections
import torchvision.transforms as T
from typing import List

from src.latent import decode_latent_representation, get_latent_model
from src.data import get_data
from src.downstream.datasets import LatentClassificationDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def process_reconstruction(
    rank: int,
    world_size: int,
    file_list: str,
    model: torch.nn.Module,
    config,
    latents_dir: str,
    save_dir: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    setup(rank, world_size, config.master_port)
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model.eval()

    TARGET_STD = 0.5
    mean = mean.to(device).view(1, -1, 1, 1).to(next(model.parameters()).dtype)
    std = std.to(device).view(1, -1, 1, 1).to(next(model.parameters()).dtype)

    dataset = LatentClassificationDataset(latents_dir, file_list, split=None, return_rel_path=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        sampler=sampler, 
        num_workers=config.num_workers,
        pin_memory=True
    )

    iterator = dataloader
    if rank == 0:
        iterator = tqdm(dataloader, desc="Reconstructing")

    with torch.no_grad():
        for latents, labels, rel_paths in iterator:
            cur_latents = latents.to(device, non_blocking=True).to(next(model.parameters()).dtype)
            
            # Denormalize
            cur_latents = (cur_latents * (std / TARGET_STD)) + mean
            
            # Decode
            reconstructed = decode_latent_representation(cur_latents, model) # [B, 3, H, W]
            
            reconstructed = (reconstructed + 1) / 2
            reconstructed = reconstructed.clamp(0, 1)
            
            for i, rel_path in enumerate(rel_paths):
                img_tensor = reconstructed[i].cpu()
                img = T.ToPILImage()(img_tensor)
                
                img_save_path = os.path.join(save_dir, rel_path)
                if not any(img_save_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                    img_save_path += ".png"
                
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                img.save(img_save_path)

    cleanup()

def run(config) -> None:
    world_size = torch.cuda.device_count()

    if not config.filelist.endswith(".csv"):
        raise ValueError("Only CSV files are supported as input.")
    
    # Load stats
    if not os.path.exists(config.mean_path) or not os.path.exists(config.std_path):
        raise FileNotFoundError(f"Stats files not found at {config.mean_path} or {config.std_path}")
    
    mean = torch.load(config.mean_path, map_location="cpu", weights_only=True)
    std = torch.load(config.std_path, map_location="cpu", weights_only=True)

    # Output directory for reconstructed images
    os.makedirs(config.save_dir, exist_ok=True)
    print(f"Reconstructing images to {config.save_dir}")

    from src.data import get_data
    file_list, _ = get_data(config)
    last_rel_path = file_list[-1]
    last_img_path = os.path.join(config.save_dir, last_rel_path)
    
    if not any(last_img_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
        last_img_path += ".png"
    
    if os.path.exists(last_img_path):
        print(f"Skipping compute_reconstruction because last image {last_img_path} already exists.")
        return

    # Model
    ckpt_path = getattr(config, "ckpt_path", None)
    model = get_latent_model(path=config.vae_path, modality=config.modality, ckpt_path=ckpt_path)

    # Spawn workers
    mp.spawn(
        process_reconstruction,
        args=(
            world_size,
            config.filelist,
            model,
            config,
            config.latents_dir,
            config.save_dir,
            mean,
            std,
        ),
        nprocs=world_size,
        join=True,
    )

def get_args():
    parser = argparse.ArgumentParser(description="Reconstruct images from latents.")
    parser.add_argument("--latents_dir", type=str, required=True, help="Directory where .pt latents are stored")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save reconstructed images")
    parser.add_argument("--filelist", type=str, required=True, help="CSV file with paths to data")
    parser.add_argument("--mean_path", type=str, required=True, help="Path to channel_mean.pt")
    parser.add_argument("--std_path", type=str, required=True, help="Path to channel_std.pt")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for decoding")
    parser.add_argument("--vae_path", type=str, default="black-forest-labs/FLUX.2-dev", help="Path or repo ID for the VAE model")
    parser.add_argument("--modality", type=str, default="xray", help="Modality for MedVAE (xray, ct, mri)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--master_port", type=int, default=12345)
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to finetuned model checkpoint (model.pt from medvae_finetune)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = ml_collections.ConfigDict(vars(args)) 
    run(config)
