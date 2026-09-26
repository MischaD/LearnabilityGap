import os
import shutil

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedGroupKFold

from src.downstream.datasets import *
from src.downstream.utils_noise_cond import *
from src.downstream.losses import *
from src.latent import get_latent_model

# --- NOISE CONDITIONING MODULES ---
# Borrowed from EDM-2 for generating the time embedding
def normalize(x, dim=None, eps=1e-4):
    if dim == None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=[]):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
        w = normalize(w)
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

class NoiseConditionedConvNeXt(nn.Module):
    def __init__(self, backbone, embed_dim=256, mean=None, std=None):
        super().__init__()
        self.backbone = backbone
        
        # Latent Normalization
        self.register_buffer('mean', mean) if mean is not None else setattr(self, 'mean', None)
        self.register_buffer('std', std) if std is not None else setattr(self, 'std', None)
        self.TARGET_STD = 0.5
        
        # Time Embedding
        self.time_fourier = MPFourier(embed_dim)
        self.time_proj1 = MPConv(embed_dim, embed_dim)
        self.time_proj2 = MPConv(embed_dim, embed_dim)
        
        # FiLM layers for ConvNeXt stages (Tiny has 4 stages with dimensions: 96, 192, 384, 768)
        # In ConvNeXt features, the stages are typically at indices 1, 3, 5, 7.
        self.film_projs = nn.ModuleList([
            nn.Linear(embed_dim, 96 * 2),
            nn.Linear(embed_dim, 192 * 2),
            nn.Linear(embed_dim, 384 * 2),
            nn.Linear(embed_dim, 768 * 2)
        ])
        
        # Zero-initialize FiLM projections so they start as an identity mapping
        for proj in self.film_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def encode_time(self, sigma):
        # Log scaling of sigma according to EDM
        c_noise = sigma.flatten().log() / 4
        emb = self.time_fourier(c_noise)
        emb = mp_silu(self.time_proj1(emb))
        emb = self.time_proj2(emb)
        return emb

    def forward(self, x, sigma):
        # Normalization
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) * (self.TARGET_STD / self.std.clamp(min=1e-12))
            
        # Noise Injection during training or if a specific sigma is passed
        if sigma is not None:
            noise = torch.randn_like(x)
            x = x + noise * sigma.view(-1, 1, 1, 1)
        else:
            # Pass a very small sigma instead of 0 to avoid log(0) = -inf
            sigma = torch.full((x.shape[0],), 1e-5, device=x.device)
            
        emb = self.encode_time(sigma)
        
        # Evaluate backbone layer by layer and inject FiLM
        # ConvNeXt features: 0 (stem), 1 (stage1), 2 (down1), 3 (stage2), 4 (down2), 5 (stage3), 6 (down3), 7 (stage4)
        stage_indices = [1, 3, 5, 7]
        film_idx = 0
        
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in stage_indices:
                film_params = self.film_projs[film_idx](emb) # [B, C*2]
                scale, shift = film_params.chunk(2, dim=1)
                # scale and shift are [B, C], x is [B, C, H, W]
                x = x * (1 + scale.view(*scale.shape, 1, 1)) + shift.view(*shift.shape, 1, 1)
                film_idx += 1
                
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier(x)
        return x
# ----------------------------------

def save_setting_summary(args, model_dir, n_classes, classes):
    summary_path = os.path.join(model_dir, 'setting_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Training Setting Summary\n")
        f.write("========================\n\n")
        f.write(f"Model Directory: {model_dir}\n")
        f.write(f"Model Name: {args.model_name}\n")
        f.write(f"Number of Classes: {n_classes}\n")
        f.write(f"Classes: {', '.join(classes)}\n")
        f.write(f"Is Latent Mode: {args.is_latent}\n")
        f.write(f"Use Encoder: {args.use_encoder}\n")
        f.write(f"Mask Ratio: {args.mask_ratio}\n")
        f.write(f"Mask Mode: {args.mask_mode}\n\n")
        
        f.write("Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
        
        f.write("\nEnvironment Info:\n")
        f.write(f"  PyTorch Version: {torch.__version__}\n")
        f.write(f"  CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"  Device: {torch.cuda.get_device_name(0)}\n")

def perform_crossfold(filelist: str, fold_num: int): 
    """ filelist: path to filelist
    foldnum: int from 0 to 4 indicating the fold. 
    """
    seed = 0 
    # from a local generator, create five folds of the current dataset. 
    # The folds are selected to not have patient leakage ("id" column in filelist) 
    # and all fold are roughly equally distributed according to classes 
    # overwrites the datafarame in filelist. 
    # returns nothing
    df = pd.read_csv(filelist, low_memory=False)
    
    # Identify class columns (metadata to exclude)
    metadata_columns = {"path", "id", "Split", "Unnamed: 0", "subject_id", "study_id", "dicom_id", "impression", "image", "Finding Labels", "FileName", "index"}
    class_columns = sorted([c for c in df.columns if c not in metadata_columns])
    
    # Take the class with max value for stratification
    # We assume each ID mostly belongs to one class for stratification purposes
    labels = df[class_columns].idxmax(axis=1)
    
    if 'id' in df.columns:
        groups = df['id']
    else:
        print("Warning: 'id' column not found for group-wise splitting. Using index as groups (no group leakage protection).")
        groups = df.index
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    
    # Generate all folds
    folds = list(sgkf.split(df, labels, groups=groups))
    
    # Assign:
    # Test = fold_num
    # Val = (fold_num + 1) % 5
    # Train = rest
    
    test_idx = folds[fold_num][1]
    val_idx = folds[(fold_num + 1) % 5][1]
    
    # Set splits
    df['Split'] = 'TRAIN'
    df.loc[test_idx, 'Split'] = 'TEST'
    df.loc[val_idx, 'Split'] = 'VAL'
    
    print(f"Fold {fold_num} split: TEST={len(test_idx)}, VAL={len(val_idx)}, TRAIN={len(df)-len(test_idx)-len(val_idx)}")
    
    df.to_csv(filelist, index=False)


def main(args):
    # Set model/output directory name
    MODEL_NAME = 'cxr-lt'
    MODEL_NAME += f'_{args.model_name}'
    MODEL_NAME += f'_rand' if args.rand_init else ''
    MODEL_NAME += f'_bal-mixup-{args.mixup_alpha}' if args.bal_mixup else ''
    MODEL_NAME += f'_mixup-{args.mixup_alpha}' if args.mixup else ''
    MODEL_NAME += f'_decoupling-{args.decoupling_method}' if args.decoupling_method != '' else ''
    MODEL_NAME += f'_rw-{args.rw_method}' if args.rw_method != '' else ''
    MODEL_NAME += f'_{args.loss}'
    MODEL_NAME += '-drw' if args.drw else ''
    MODEL_NAME += 'reloadbest' if args.drw_reloadbest else ''
    MODEL_NAME += f'_cb-beta-{args.cb_beta}' if args.rw_method == 'cb' else ''
    MODEL_NAME += f'_fl-gamma-{args.fl_gamma}' if args.loss == 'focal' else ''
    # Use format to ensure consistent decimal representation for learning rate
    lr_str = f"{args.lr:.4f}" if args.lr >= 1e-4 else f"{args.lr:g}"
    MODEL_NAME += f'_lr-{lr_str}'
    MODEL_NAME += f'_bs-{args.batch_size}'
    MODEL_NAME += f'_fold_{args.fold}' if args.do_crossfold else ''

    model_dir = os.path.join(args.out_dir, MODEL_NAME)
    
    # IDEMPOTENCY CHECK: Skip if test_summary.txt already exists
    if os.path.exists(os.path.join(model_dir, 'test_summary.txt')):
        print(f"=== [IDEMPOTENCY] Skipping {MODEL_NAME} as test_summary.txt already exists. ===")
        return

    print(f"Training Model: {MODEL_NAME}")
    print(f"Output Directory: {model_dir}")

    # Create output directory for model (and delete if already exists but incomplete)
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.isdir(model_dir):
        print(f"Removing existing incomplete directory: {model_dir}")
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Prepare crossfold validation 
    filelist_to_use = args.filelist
    if args.do_crossfold:
        # Create a temporary filelist for this fold securely inside its output directory
        dataset_name = os.path.basename(args.filelist).split('.')[0]
        new_filelist = os.path.join(model_dir, f"{dataset_name}_fold_{args.fold}.csv")
        shutil.copy(args.filelist, new_filelist)
        filelist_to_use = new_filelist
        perform_crossfold(new_filelist, args.fold)


    # Create datasets + loaders
    std_path = args.mean_path.replace('_mean.pt', '_std.pt') if args.mean_path else None
    train_dataset = ClassificationDataset(data_dir=args.data_dir, filelist=filelist_to_use, split='TRAIN', is_latent=args.is_latent, mask_ratio=args.mask_ratio, mean_path=args.mean_path, std_path=std_path, vae_mode=args.use_encoder, mask_mode=args.mask_mode)
    val_dataset = ClassificationDataset(data_dir=args.data_dir, filelist=filelist_to_use, split='VAL', is_latent=args.is_latent, mask_ratio=0.0, mean_path=args.mean_path, std_path=std_path, vae_mode=args.use_encoder, mask_mode=args.mask_mode)
    test_dataset = ClassificationDataset(data_dir=args.data_dir, filelist=filelist_to_use, split='TEST', is_latent=args.is_latent, mask_ratio=0.0, mean_path=args.mean_path, std_path=std_path, vae_mode=args.use_encoder, mask_mode=args.mask_mode)

    N_CLASSES = len(train_dataset.CLASSES)
    print(f"Detected {N_CLASSES} classes: {train_dataset.CLASSES}")

    # Save summary of settings
    save_setting_summary(args, model_dir, N_CLASSES, train_dataset.CLASSES)

    if args.bal_mixup:
        cls_weights = [len(train_dataset) / cls_count for cls_count in train_dataset.cls_num_list]
        instance_weights = [cls_weights[label] for label in train_dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(instance_weights), len(train_dataset))
        bal_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, sampler=sampler)

        imbal_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

        train_loader = ComboLoader([imbal_train_loader, bal_train_loader])
    elif args.decoupling_method == 'cRT':
        cls_weights = [len(train_dataset) / cls_count for cls_count in train_dataset.cls_num_list]
        instance_weights = [cls_weights[label] for label in train_dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(instance_weights), len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=val_worker_init_fn)

    # Create csv documenting training history
    history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'balanced_acc', 'mcc', 'auroc'])
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    # Set device
    device = torch.device('cuda:0')

    # Instantiate model
    if args.is_latent:
        sample_x, _ = train_dataset[0]
        in_channels, d, h = sample_x.shape
        if d != h:
            raise ValueError(f"Latent must be C x D x H (D == H), got {sample_x.shape}")
        print(f"Detected latent channels: {in_channels}")
    elif args.use_encoder:
        # Load encoder early to determine in_channels
        print(f"Loading encoder from {args.vae_path} to determine latent shape")
        encoder = get_latent_model(path=args.vae_path, device=device, modality=args.modality)
        sample_img, _ = train_dataset[0]
        with torch.no_grad():
            vae_dtype = next(encoder.parameters()).dtype
            if hasattr(encoder, "model_name") and "medvae" in encoder.model_name:
                sample_latent = encoder.encode(sample_img.unsqueeze(0).to(device).to(vae_dtype))
            else:
                sample_latent = encoder.encode(sample_img.unsqueeze(0).to(device).to(vae_dtype)).latent_dist.mode()
        in_channels, d, h = sample_latent.shape[1:]
        if d != h:
            raise ValueError(f"Encoder latent must be C x D x H (D == H), got {sample_latent.shape[1:]}")
        print(f"Detected encoder latent channels: {in_channels}")
    else:
        in_channels = 3
    
    if args.model_name == 'ConvNeXt-Tiny':
        model = models.convnext_tiny(pretrained=(not args.rand_init))
        
        if in_channels != 3:
            print(f"Overwriting first Conv layer with in_channels={in_channels}")
            # For ConvNeXt, the first layer is in features.0
            original_conv = model.features[0][0]  # Get the Conv2d layer
            model.features[0][0] = nn.Conv2d(
                in_channels, 
                original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding, 
                bias=original_conv.bias is not None
            )
        
        if args.model_path is not None:
            model_path = args.model_path
            # Support templating for fold and dataset
            if "{fold}" in model_path and hasattr(args, 'fold'):
                model_path = model_path.replace("{fold}", str(args.fold))
            if "{ds}" in model_path:
                # We can try to infer dataset from the filelist or out_dir
                ds_name = os.path.basename(args.filelist).split('.')[0]
                model_path = model_path.replace("{ds}", ds_name)
            
            print(f"Loading pretrained weights from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('weights', checkpoint)
            # Filter out classifier head weights to avoid shape mismatch
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.2.')}
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights with message: {msg}")

        model.classifier[2] = nn.Linear(model.classifier[2].in_features, N_CLASSES)
    else:
        print(f"Using Model: {args.model_name}")
        model = getattr(models, args.model_name)(pretrained=(not args.rand_init))
        #model = torchvision.models.resnet._resnet(torchvision.models.resnet.Bottleneck, [1, 3, 6, 3], None, True)
        #model.layer1 = nn.Identity()
        #model.layer2 = torchvision.model.resenet._make_layer(torchvision.model.resnet.BasicBlock, 64, )
        
        if in_channels != 3:
            print(f"Overwriting first Conv layer with in_channels={in_channels}")
            original_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
        if args.model_path is not None:
            print(f"Loading pretrained weights from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location='cpu')
            state_dict = checkpoint.get('weights', checkpoint)
            # Filter out classifier head weights to avoid shape mismatch
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights with message: {msg}")

        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)


    if args.decoupling_method == 'tau_norm':
        msg = model.load_state_dict(torch.load(args.decoupling_weights, map_location='cpu')['weights'])
        print(f'Loaded weights from {args.decoupling_weights} with message: {msg}')

        model.fc.bias.data = torch.zeros_like(model.fc.bias.data)
        fc_weights = model.fc.weight.data.clone()

        weight_norms = torch.norm(fc_weights, 2, 1)

        model.fc.weight.data = torch.stack([fc_weights[i] / torch.pow(weight_norms[i], -4) for i in range(N_CLASSES)], dim=0)
    elif args.decoupling_method == 'cRT':
        msg = model.load_state_dict(torch.load(args.decoupling_weights, map_location='cpu')['weights'])
        print(f'Loaded weights from {args.decoupling_weights} with message: {msg}')

        model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)  # re-initialize classifier head

    mean, std = None, None
    if args.mean_path and os.path.exists(args.mean_path):
        mean = torch.load(args.mean_path, map_location='cpu', weights_only=True).view(1, -1, 1, 1).to(device)
        std_path = args.mean_path.replace('_mean.pt', '_std.pt')
        print(f"Normalization path loaded: {std_path}")
        if os.path.exists(std_path):
            std = torch.load(std_path, map_location='cpu', weights_only=True).view(1, -1, 1, 1).to(device)
            
    # Wrap model in NoiseConditionedConvNeXt
    model = NoiseConditionedConvNeXt(model, mean=mean, std=std)

    model = model.to(device)        

    # Set loss and weighting method
    if args.rw_method == 'sklearn':
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.labels), y=np.array(train_dataset.labels))
        weights = torch.Tensor(weights).to(device)
    elif args.rw_method == 'cb':
        weights = get_CB_weights(samples_per_cls=train_dataset.cls_num_list, beta=args.cb_beta)
        weights = torch.Tensor(weights).to(device)
    else:
        weights = None

    if weights is None:
        print('No class reweighting')
    else:
        print(f'Class weights with rw_method {args.rw_method}:')
        for i, c in enumerate(train_dataset.CLASSES):
            print(f'\t{c}: {weights[i]}')

    loss_fxn = get_loss(args, None if args.drw else weights, train_dataset)

    # Set optimizer
    if args.decoupling_method != '':
        # Since we are using NoiseConditionedConvNeXt, we access the linear head via model.backbone.classifier
        # For ConvNeXt, the classifier head is the 3rd element in the classifier module
        target_model = model.backbone.classifier[2] if hasattr(model.backbone, 'classifier') else model.backbone.fc
        optimizer = torch.optim.Adam(target_model.parameters(), lr=args.lr)    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train with early stopping
    if args.decoupling_method != 'tau_norm':
        epoch = 1
        early_stopping_dict = {'best_acc': 0., 'epochs_no_improve': 0}
        best_model_wts = None
        while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
            if args.bal_mixup:
                history = bal_mixup_train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES, mixup_alpha=args.mixup_alpha)    
            else:
                history = train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES, mixup=args.mixup, mixup_alpha=args.mixup_alpha)
            history, early_stopping_dict, best_model_wts = validate(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, classes=val_dataset.CLASSES)

            if args.drw and epoch == args.drw_epoch:
                if args.drw_reloadbest:
                    best_ckpt = torch.load(os.path.join(model_dir, 'best.pt'), weights_only=True, map_location="cpu")
                    print(f'--- DRW: Reloading best model weights from epoch {epoch - early_stopping_dict["epochs_no_improve"]} (acc: {round(early_stopping_dict["best_acc"], 3)}) ---')
                    model.load_state_dict(best_ckpt["weights"])
                    optimizer.load_state_dict(best_ckpt["optimizer"])
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1  # anneal LR
                loss_fxn = get_loss(args, weights, train_dataset)  # get class-weighted loss
                early_stopping_dict['epochs_no_improve'] = 0  # reset patience

            epoch += 1
    else:
        best_model_wts = model.state_dict()
    


    # Evaluate on imbalanced test set
    evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts)

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/ssd1/greg/NIH_CXR/images', type=str)
    parser.add_argument('--filelist', required=True, type=str)
    parser.add_argument('--out_dir', default='results/', type=str, help="path to directory where results and model weights will be saved")
    parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal', 'ldam'])
    parser.add_argument('--drw', action='store_true', default=False)
    parser.add_argument('--drw_epoch', default=10, type=int, help="epoch at which to start DRW")
    parser.add_argument('--drw_reloadbest', action='store_true', default=False, help="reload best model weights before starting DRW re-weighting")
    parser.add_argument('--rw_method', default='', choices=['', 'sklearn', 'cb'])
    parser.add_argument('--cb_beta', default=0.9999, type=float)
    parser.add_argument('--fl_gamma', default=2., type=float)
    parser.add_argument('--bal_mixup', action='store_true', default=False)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', default=0.2, type=float)
    parser.add_argument('--decoupling_method', default='', choices=['', 'cRT', 'tau_norm'], type=str)
    parser.add_argument('--decoupling_weights', type=str)
    parser.add_argument('--model_name', default='resnet50', type=str, help="CNN backbone to use (used if --network not ConvNeXt-Tiny)")
    parser.add_argument('--model_path', default=None, type=str, help="path to pre-trained model weights")
    parser.add_argument('--max_epochs', default=60, type=int, help="maximum number of epochs to train")
    parser.add_argument('--batch_size', default=256, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=15, type=int, help="early stopping 'patience' during training")
    parser.add_argument('--mask_ratio', default=0.0, type=float, help="Ratio of random masking to apply to latents (0.0 to 1.0)")
    parser.add_argument('--mask_mode', default='default', type=str, choices=['default', 'cw'], help="Masking mode: 'default' (pixel-wise) or 'cw' (channel-wise)")
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--n_TTA', default=0, type=int, help="number of augmented copies to use during test-time augmentation (TTA), default 0")
    parser.add_argument('--is_latent', action='store_true', default=False)
    parser.add_argument('--mean_path', type=str, default=None, help="Path to channel_mean.pt for latent normalization (std path is derived)")
    parser.add_argument('--use_encoder', action='store_true', default=False, help="Use a frozen encoder to process images on the fly")
    parser.add_argument('--vae_path', type=str, default="black-forest-labs/FLUX.2-dev", help="Path or repo ID for the VAE encoder")
    parser.add_argument('--modality', type=str, default="xray", help="Modality for MedVAE (xray, ct, mri)")
    parser.add_argument('--seed', default=0, type=int, help="set random seed")
    parser.add_argument('--do_crossfold', action='store_true', default=False)

    args = parser.parse_args()

    if args.is_latent and args.use_encoder:
        raise ValueError("Cannot use both --is_latent and --use_encoder. Choose one. "
                         "Use --is_latent if you have precomputed latents. "
                         "Use --use_encoder if you have PNG images and want to encode them on the fly.")

    print(args)
    if args.do_crossfold:
        print("Performing fivefold crossvalidation")
        for fold in range(5): 
            args.seed = fold
            setattr(args, "fold", fold)
            main(args)
    else:
        main(args)
        # 

