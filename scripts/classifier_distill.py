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

import tqdm
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from src.downstream.utils import compute_auc

import cv2
class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, latent_dir, filelist, split, mean_path=None, std_path=None):
        self.data_dir = data_dir
        self.latent_dir = latent_dir
        self.split = split
        self.label_df = pd.read_csv(filelist)
        if self.split:
            self.label_df = self.label_df[self.label_df['Split'] == self.split]
        metadata_columns = {"path", "id", "Split", "Unnamed: 0", "subject_id", "study_id", "dicom_id", "impression", "image", "Finding Labels", "FileName", "index"}
        self.CLASSES = sorted([c for c in self.label_df.columns if c not in metadata_columns])
        
        self.path_col = 'path' if 'path' in self.label_df.columns else 'id'
        self.rel_paths = self.label_df[self.path_col].values.tolist()
        self.img_paths = [os.path.join(self.data_dir, p) for p in self.rel_paths]
        self.latent_paths = [os.path.join(self.latent_dir, p) for p in self.rel_paths]
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values
        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        # Image transform for teacher
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.mean = None
        self.std = None
        if mean_path and os.path.exists(mean_path):
            self.mean = torch.load(mean_path, map_location='cpu', weights_only=True).view(-1, 1, 1)
        if std_path and os.path.exists(std_path):
            self.std = torch.load(std_path, map_location='cpu', weights_only=True).view(-1, 1, 1)
        self.TARGET_STD = 0.5

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        latent_path = self.latent_paths[idx]
        
        latent = torch.load(latent_path + ".pt", weights_only=True)
        if self.mean is not None and self.std is not None:
            latent = (latent - self.mean) * (self.TARGET_STD / self.std.clamp(min=1e-12))
            
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = self.img_transform(img)

        y = np.array(self.labels[idx])
        return latent.float(), img.float(), torch.from_numpy(y).long()

def train_distill(student, teacher, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, alpha):
    student.train()
    teacher.eval()
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    running_loss = 0.
    y_true, y_hat = [], []
    mse_loss_fn = torch.nn.MSELoss()
    for i, (latent, img, y) in pbar:
        latent, img, y = latent.to(device), img.to(device), y.to(device)
        with torch.no_grad():
            teacher_logits = teacher(img)
        student_logits = student(latent)
        
        ce_loss = loss_fxn(student_logits, y)
        distill_loss = mse_loss_fn(student_logits, teacher_logits)
        loss = alpha * ce_loss + (1 - alpha) * distill_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        y_hat.append(student_logits.softmax(dim=1).detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())
        pbar.set_postfix({'loss': running_loss / (i + 1)})
        
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)
    auc = compute_auc(y_true, y_hat)
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))
    print('Balanced Accuracy:', round(b_acc, 3), '|', 'MCC:', round(mcc, 3), '|', 'AUC:', round(auc, 3))
    
    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (i + 1), b_acc, mcc, auc]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)
    return pd.concat([history, current_metrics], ignore_index=True)

def validate_distill(student, teacher, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, alpha):
    student.eval()
    teacher.eval()
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')
    running_loss = 0.
    y_true, y_hat = [], []
    mse_loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for i, (latent, img, y) in pbar:
            latent, img, y = latent.to(device), img.to(device), y.to(device)
            teacher_logits = teacher(img)
            student_logits = student(latent)
            
            ce_loss = loss_fxn(student_logits, y)
            distill_loss = mse_loss_fn(student_logits, teacher_logits)
            loss = alpha * ce_loss + (1 - alpha) * distill_loss
            
            running_loss += loss.item()
            y_hat.append(student_logits.softmax(dim=1).detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())
            pbar.set_postfix({'loss': running_loss / (i + 1)})
            
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)
    auc = compute_auc(y_true, y_hat)
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))
    print('[VAL] Balanced Accuracy:', round(b_acc, 3), '|', 'MCC:', round(mcc, 3), '|', 'AUC:', round(auc, 3))
    
    current_metrics = pd.DataFrame([[epoch, 'val', running_loss / (i + 1), b_acc, mcc, auc]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)
    
    if b_acc > early_stopping_dict['best_acc']:
        print(f'--- EARLY STOPPING: Accuracy has improved from {round(early_stopping_dict["best_acc"], 3)} to {round(b_acc, 3)}! Saving weights. ---')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_acc'] = b_acc
        from copy import deepcopy
        best_model_wts = deepcopy(student.state_dict())
        torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, 'best.pt'))
    else:
        print(f'--- EARLY STOPPING: Accuracy has not improved from {round(early_stopping_dict["best_acc"], 3)} ---')
        early_stopping_dict['epochs_no_improve'] += 1
        
    torch.save({'weights': student.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, 'latest.pt'))
    return pd.concat([history, current_metrics], ignore_index=True), early_stopping_dict, best_model_wts

def evaluate_distill(model, device, dataset, batch_size, model_dir, weights):
    model.load_state_dict(weights)
    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc='[TEST] EVALUATION')
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (latent, img, y) in pbar:
            latent = latent.to(device)
            out = model(latent)
            y_hat.append(out.softmax(dim=1).detach().cpu().numpy())
            y_true.append(y.numpy())
    
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    auc = compute_auc(y_true, y_hat)
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))
    
    conf_mat = confusion_matrix(y_true, y_hat.argmax(axis=1))
    accuracies = conf_mat.diagonal() / conf_mat.sum(axis=1)
    
    print(f'[TEST] Balanced Accuracy: {round(b_acc, 3)} | MCC: {round(mcc, 3)} | AUC: {round(auc, 3)}')
    
    pred_df = pd.DataFrame(y_hat, columns=dataset.CLASSES)
    pred_df.to_csv(os.path.join(model_dir, 'test_pred.csv'), index=False)
    
    # Save true labels as well
    true_df = pd.DataFrame(y_true, columns=['label'])
    true_df.to_csv(os.path.join(model_dir, 'test_true.csv'), index=False)
    
    # Create summary text file describing final performance in format expected by evaluation scripts
    summary = f'Balanced Accuracy: {round(b_acc, 4)}\n'
    summary += f'Matthews Correlation Coefficient: {round(mcc, 4)}\n'
    summary += f'Mean AUC: {round(auc, 4)}\n\n'
    
    summary += 'Class:| Accuracy\n'
    for i, c in enumerate(dataset.CLASSES):
        summary += f'{c}:| {round(accuracies[i], 4)}\n'
    
    with open(os.path.join(model_dir, 'test_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"Results saved to {model_dir}")


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
    
    # IDEMPOTENCY CHECK: Skip if test_summary.txt already exists (only if not test_only)
    if not args.test_only and os.path.exists(os.path.join(model_dir, 'test_summary.txt')):
        print(f"=== [IDEMPOTENCY] Skipping {MODEL_NAME} as test_summary.txt already exists. ===")
        return

    print(f"Training Model: {MODEL_NAME}")
    print(f"Output Directory: {model_dir}")

    # Create output directory for model (and delete if already exists but incomplete)
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.test_only and os.path.isdir(model_dir):
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
    train_dataset = DistillationDataset(data_dir=args.data_dir, latent_dir=args.latent_dir, filelist=filelist_to_use, split='TRAIN', mean_path=args.mean_path, std_path=std_path)
    val_dataset = DistillationDataset(data_dir=args.data_dir, latent_dir=args.latent_dir, filelist=filelist_to_use, split='VAL', mean_path=args.mean_path, std_path=std_path)
    test_dataset = DistillationDataset(data_dir=args.data_dir, latent_dir=args.latent_dir, filelist=filelist_to_use, split='TEST', mean_path=args.mean_path, std_path=std_path)

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
    sample_latent, sample_img, sample_y = train_dataset[0]
    in_channels, d, h = sample_latent.shape
    if d != h:
        raise ValueError(f"Latent must be C x D x H (D == H), got {sample_latent.shape}")
    print(f"Detected latent channels: {in_channels}")
    
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
            print(f"Loading pretrained weights from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location='cpu')
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


    print(f"Loading Teacher Model: {args.teacher_model}")
    if args.teacher_model == 'ConvNeXt-Tiny':
        teacher = models.convnext_tiny(pretrained=True)
        if hasattr(teacher.classifier[2], "out_features") and teacher.classifier[2].out_features != N_CLASSES:
            print("Reinitializing teacher classifier head due to mismatched N_CLASSES")
            teacher.classifier[2] = nn.Linear(teacher.classifier[2].in_features, N_CLASSES)
    else:
        teacher = getattr(models, args.teacher_model)(pretrained=True)
        if hasattr(teacher, "fc") and getattr(teacher.fc, "out_features", None) != N_CLASSES:
            print("Reinitializing teacher classifier head due to mismatched N_CLASSES")
            teacher.fc = nn.Linear(teacher.fc.in_features, N_CLASSES)

    if args.teacher_weights_dir is not None:
        import glob
        if args.do_crossfold:
            search_pattern = os.path.join(args.teacher_weights_dir, f"*_fold_{args.fold}", "best.pt")
            matched_paths = glob.glob(search_pattern)
            if not matched_paths:
                raise FileNotFoundError(f"No teacher best.pt found matching {search_pattern}")
            teacher_path = matched_paths[0]
        else:
            teacher_path = args.teacher_weights_dir
            
        print(f"Loading teacher weights from {teacher_path}")
        checkpoint = torch.load(teacher_path, map_location='cpu')
        state_dict = checkpoint.get('weights', checkpoint)
        msg = teacher.load_state_dict(state_dict, strict=False)
        print(f"Loaded teacher weights with message: {msg}")

    teacher = teacher.to(device)
    teacher.eval()

    # Train with early stopping
    if not args.test_only:
        if args.decoupling_method != 'tau_norm':
            epoch = 1
            early_stopping_dict = {'best_acc': 0., 'epochs_no_improve': 0}
            best_model_wts = None
            
            while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
                history = train_distill(model, teacher, device, loss_fxn, optimizer, train_loader, history, epoch, model_dir, args.alpha)
                history, early_stopping_dict, best_model_wts = validate_distill(model, teacher, device, loss_fxn, optimizer, val_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, args.alpha)
                
                if args.drw and epoch == args.drw_epoch:
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.1
                    loss_fxn = get_loss(args, weights, train_dataset)
                    early_stopping_dict['epochs_no_improve'] = 0

                epoch += 1

        else:
            best_model_wts = model.state_dict()
    else:
        print(f"Test only mode: loading best weights from {os.path.join(model_dir, 'best.pt')}")
        best_model_wts = torch.load(os.path.join(model_dir, 'best.pt'), map_location='cpu')['weights']
    


    # Evaluate on imbalanced test set
    evaluate_distill(model, device, test_dataset, args.batch_size, model_dir, best_model_wts)

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/ssd1/greg/NIH_CXR/images', type=str)
    parser.add_argument('--latent_dir', required=True, type=str, help='directory containing latents')
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
    parser.add_argument('--alpha', default=0.5, type=float, help='KD tradeoff; 1.0=CE only')
    parser.add_argument('--teacher_model', default='ConvNeXt-Tiny', type=str)
    parser.add_argument('--teacher_weights_dir', default=None, type=str, help='Directory containing fold subdirs with best.pt')
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
    parser.add_argument('--test_only', action='store_true', default=False)

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

