import torch
import torchvision.transforms as transforms
import torchvision.io as io
from diffusers import AutoencoderKL
from einops import repeat


def compute_latent_representation(input_tensor, model, batch_size):
    # L x C x H x W
    latent_representations = []
    L = input_tensor.shape[0]

    for i in range(0, L, batch_size):
        end = min(i + batch_size, L)
        batch = input_tensor[i:end]
        
        with torch.no_grad():
            if hasattr(model, "model_name") and "medvae" in model.model_name:
                # MVAE model
                latent = model.encode(batch)
            else:
                # AutoencoderKL model
                latent = model.encode(batch).latent_dist.mode()
        
        latent_representations.append(latent)
    
    latent_representations = torch.cat(latent_representations, dim=0)
    return latent_representations

def decode_latent_representation(latent_representations, model):
    with torch.no_grad():
        if hasattr(model, "model_name") and "medvae" in model.model_name:
            decoded_video = model.decode(latent_representations)
        else:
            decoded_video = model.decode(latent_representations).sample
    
    return decoded_video

def get_latent_model(path, device="cuda", modality="xray", ckpt_path=None):
    if path.startswith("medvae_"):
        from medvae import MVAE
        model = MVAE(model_name=path, modality=modality)

        if ckpt_path is not None:
            print(f"Loading finetuned MedVAE weights from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()
                
            cleaned = {}
            for k, v in state_dict.items():
                new_k = k
                for prefix in ["module.", "_orig_mod."]:
                    while new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                
                if path.startswith("medvae_") and not new_k.startswith("model."):
                    if any(c in new_k for c in ["encoder", "decoder", "channel_ds", "channel_proj", "quant_conv", "post_quant_conv"]):
                        new_k = f"model.{new_k}"
                     
                cleaned[new_k] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"Loaded finetuned weights: {len(missing)} missing, {len(unexpected)} unexpected keys")
            if missing:
                print(f"  Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"  Unexpected: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

        model = model.to(device)
        model.eval()
        return model

    # Load the VQ-VAE/Flux model
    try:
        model = AutoencoderKL.from_pretrained(
            path, 
            subfolder="vae",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            token=True
        )
    except Exception as e:
        print(f"Failed to load with subfolder='vae': {e}. Retrying root...")
        model = AutoencoderKL.from_pretrained(path, token=True)
    
    model = model.to(device)
    model.eval()
    return model