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
from src.downstream.utils import *
from src.downstream.losses import *
from src.latent import get_latent_model

class EncodedModel(nn.Module):
    def __init__(self, encoder, classifier, mean=None, std=None, mask_ratio=0.0, mask_mode="default"):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.register_buffer('mean', mean) if mean is not None else setattr(self, 'mean', None)
        self.register_buffer('std', std) if std is not None else setattr(self, 'std', None)
        self.TARGET_STD = 0.5
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            # Ensure input is in the correct dtype for VAE (usually float16 on GPU)
            vae_dtype = next(self.encoder.parameters()).dtype
            # x: [B, 3, H, W]
            if hasattr(self.encoder, "model_name") and "medvae" in self.encoder.model_name:
                latents = self.encoder.encode(x.to(vae_dtype))
            else:
                latents = self.encoder.encode(x.to(vae_dtype)).latent_dist.mode()
            
            if self.mean is not None and self.std is not None:
                latents = (latents - self.mean) * (self.TARGET_STD / self.std.clamp(min=1e-12))
            
            if self.training and self.mask_ratio > 0:
                if self.mask_mode == "default":
                    mask = torch.rand_like(latents) > self.mask_ratio
                elif self.mask_mode == "cw":
                    mask = torch.rand(latents.shape[0], latents.shape[1], 1, 1, device=latents.device) > self.mask_ratio
                latents = latents * mask
                
        # Pass latents to classifier, ensuring they are back to float32 if needed
        return self.classifier(latents.to(x.dtype))

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

def perform_crossfold(filelist: str, fold_num: int, aug_dir: str = ""): 
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
    
    if aug_dir != "":
        train_df = df[df['Split'] == 'TRAIN'].copy()
        
        path_col = None
        if 'path' in train_df.columns:
            path_col = 'path'
        elif 'id' in train_df.columns:
            path_col = 'id'
            
        if path_col:
            # We assume aug_dir is an absolute path to the generated images
            # Using os.path.join to prepend aug_dir to relative path
            train_df[path_col] = train_df[path_col].apply(lambda p: os.path.join(aug_dir, p) if not str(p).startswith('/') else p)
            
            # If the CSV has a ".pt" ending or similar (as latents might), we keep the original extension or assume Dataset handles it
            df = pd.concat([df, train_df], ignore_index=True)
            print(f"Augmentation: Added {len(train_df)} synthesized samples from {aug_dir} to TRAIN split.")

    print(f"Fold {fold_num} split: TEST={len(test_idx)}, VAL={len(val_idx)}, TRAIN={len(df)-len(test_idx)-len(val_idx)}")
    
    df.to_csv(filelist, index=False)


def _make_resnet50_backbone(in_channels: int, small_stem: bool, pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet50(weights=weights)

    if small_stem:
        pretrained_w = backbone.conv1.weight.data  # (64, 3, 7, 7)
        if in_channels != 3:
            pretrained_w = pretrained_w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
        center = pretrained_w[:, :, 2:5, 2:5].clone()
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        new_conv1.weight = nn.Parameter(center)
        stem_layers = [new_conv1, backbone.bn1, backbone.relu]  # no maxpool
    else:
        if in_channels != 3:
            avg_w = backbone.conv1.weight.data.mean(dim=1, keepdim=True)
            new_w = avg_w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
            new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv1.weight = nn.Parameter(new_w)
            backbone.conv1 = new_conv1
        stem_layers = [backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool]

    net = nn.Sequential(
        *stem_layers,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
        backbone.avgpool,
    )
    return net, 2048


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
    
    print(f"Training Model: {MODEL_NAME}")
    print(f"Output Directory: {model_dir}")

    # Create output directory for model
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Prepare crossfold validation 
    filelist_to_use = args.filelist
    if args.do_crossfold:
        # Create a temporary filelist for this fold to avoid overwriting the original
        new_filelist = os.path.join(model_dir, f"tmp_fold_{args.fold}.csv")
        shutil.copy(args.filelist, new_filelist)
        filelist_to_use = new_filelist
        perform_crossfold(new_filelist, args.fold, args.aug_dir)


    # Create datasets + loaders
    std_path = args.mean_path.replace('_mean.pt', '_std.pt') if args.mean_path else None
    train_dataset = ClassificationDataset(data_dir=args.data_dir, filelist=filelist_to_use, split='TRAIN', is_latent=args.is_latent, mask_ratio=args.mask_ratio, mean_path=args.mean_path, std_path=std_path, vae_mode=args.use_encoder, mask_mode=args.mask_mode)
    val_dataset = ClassificationDataset(data_dir=args.data_dir, filelist=filelist_to_use, split='VAL', is_latent=args.is_latent, mask_ratio=0.0, mean_path=args.mean_path, std_path=std_path, vae_mode=args.use_encoder, mask_mode=args.mask_mode)
    test_data_dir = args.test_data_dir if args.test_data_dir else args.data_dir
    test_dataset = ClassificationDataset(data_dir=test_data_dir, filelist=filelist_to_use, split='TEST', is_latent=args.is_latent, mask_ratio=0.0, mean_path=args.mean_path, std_path=std_path, vae_mode=args.use_encoder, mask_mode=args.mask_mode)

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
    
    print(f"Using Model: {args.model_name}")
    if args.model_name == 'ConvNeXt-Tiny':
        model = models.convnext_tiny(pretrained=(not args.rand_init))
    elif args.model_name == 'resnet50_small_stem':
        backbone, feat_dim = _make_resnet50_backbone(in_channels, small_stem=True, pretrained=not args.rand_init)
        model = nn.Sequential(backbone, nn.Flatten(), nn.Linear(feat_dim, N_CLASSES))
    else:
        model = getattr(models, args.model_name)(pretrained=(not args.rand_init))

        if in_channels != 3:
            print(f"Overwriting first Conv layer with in_channels={in_channels}")
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    original_conv = module
                    attrs = name.split('.')
                    parent = model
                    for attr in attrs[:-1]:
                        parent = getattr(parent, attr)
                    setattr(parent, attrs[-1], nn.Conv2d(
                        in_channels,
                        original_conv.out_channels,
                        kernel_size=original_conv.kernel_size,
                        stride=original_conv.stride,
                        padding=original_conv.padding,
                        bias=original_conv.bias is not None
                    ))
                    break

        # Find last linear to handle model weight filtering and head replacement
        last_linear_name = None
        last_linear_module = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear_name = name
                last_linear_module = module

        if args.model_path is not None:
            model_path = args.model_path
            # Support {fold} and {ds} templating (same pattern as classifier_train_noise_cond.py)
            if '{fold}' in model_path and hasattr(args, 'fold'):
                model_path = model_path.replace('{fold}', str(args.fold))
            if '{ds}' in model_path:
                ds_name = os.path.basename(args.filelist).split('.')[0]
                model_path = model_path.replace('{ds}', ds_name)
            print(f"Loading pretrained weights from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('weights', checkpoint)
            if last_linear_name is not None:
                # Filter out classifier head weights to avoid shape mismatch
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith(last_linear_name)}
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights with message: {msg}")

        if last_linear_name is not None:
            attrs = last_linear_name.split('.')
            parent = model
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, attrs[-1], nn.Linear(
                last_linear_module.in_features,
                N_CLASSES,
                bias=last_linear_module.bias is not None
            ))
        else:
            print("Warning: No nn.Linear found in model to use as classifier head.")


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

    if args.use_encoder:
        print(f"Loading encoder from {args.vae_path}")
        mean, std = None, None
        if args.mean_path and os.path.exists(args.mean_path):
            mean = torch.load(args.mean_path, map_location='cpu', weights_only=True).view(1, -1, 1, 1).to(device)
            std_path = args.mean_path.replace('_mean.pt', '_std.pt')
            if os.path.exists(std_path):
                std = torch.load(std_path, map_location='cpu', weights_only=True).view(1, -1, 1, 1).to(device)
                
        # encoder is already loaded
        model = EncodedModel(encoder, model, mean=mean, std=std, mask_ratio=args.mask_ratio, mask_mode=args.mask_mode)

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
        # If wrapped in EncodedModel, the classifier is model.classifier
        target_model = model.classifier if args.use_encoder else model
        optimizer = torch.optim.Adam(target_model.fc.parameters(), lr=args.lr)    
    else:
        # If wrapped in EncodedModel, we only want to optimize the classifier part
        trainable_params = model.classifier.parameters() if args.use_encoder else model.parameters()
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Train with early stopping
    best_pt_path = os.path.join(model_dir, 'best.pt')
    if os.path.exists(best_pt_path):
        print(f"Found existing {best_pt_path}, skipping training and directly evaluating.")
        best_ckpt = torch.load(best_pt_path, map_location='cpu')
        if 'weights' in best_ckpt:
            best_model_wts = best_ckpt['weights']
        else:
            best_model_wts = best_ckpt
    elif args.decoupling_method != 'tau_norm':
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

    if args.do_crossfold and os.path.exists(filelist_to_use):
        if 'tmp_fold' in filelist_to_use:
            os.remove(filelist_to_use)

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/ssd1/greg/NIH_CXR/images', type=str)
    parser.add_argument('--test_data_dir', default=None, type=str, help="path to directory containing test data (if different from data_dir)")
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
    parser.add_argument('--fold', default=None, type=int, help="Run a single specific fold (0-4). Used with --do_crossfold for parallel launching.")
    parser.add_argument('--aug_dir', default='', type=str, help="path to generated samples to use as augmentation on train split")

    args = parser.parse_args()

    if args.is_latent and args.use_encoder:
        raise ValueError("Cannot use both --is_latent and --use_encoder. Choose one. "
                         "Use --is_latent if you have precomputed latents. "
                         "Use --use_encoder if you have PNG images and want to encode them on the fly.")

    print(args)
    if args.do_crossfold:
        print("Performing fivefold crossvalidation")
        folds_to_run = [args.fold] if args.fold is not None else range(5)
        for fold in folds_to_run:
            args.seed = fold
            setattr(args, "fold", fold)
            main(args)
    else:
        main(args)
        # 

