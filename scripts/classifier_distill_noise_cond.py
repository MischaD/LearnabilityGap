import os
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import cv2
import tqdm

from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from src.downstream.datasets import ClassificationDataset
from src.downstream.utils import compute_auc, set_seed, worker_init_fn, val_worker_init_fn
from src.downstream.losses import get_loss, get_CB_weights
from src.latent import get_latent_model
from sklearn.preprocessing import LabelBinarizer

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
        
        self.register_buffer('mean', mean) if mean is not None else setattr(self, 'mean', None)
        self.register_buffer('std', std) if std is not None else setattr(self, 'std', None)
        self.TARGET_STD = 0.5
        
        self.time_fourier = MPFourier(embed_dim)
        self.time_proj1 = MPConv(embed_dim, embed_dim)
        self.time_proj2 = MPConv(embed_dim, embed_dim)
        
        self.film_projs = nn.ModuleList([
            nn.Linear(embed_dim, 96 * 2),
            nn.Linear(embed_dim, 192 * 2),
            nn.Linear(embed_dim, 384 * 2),
            nn.Linear(embed_dim, 768 * 2)
        ])
        
        for proj in self.film_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def encode_time(self, sigma):
        c_noise = sigma.flatten().log() / 4
        emb = self.time_fourier(c_noise)
        emb = mp_silu(self.time_proj1(emb))
        emb = self.time_proj2(emb)
        return emb

    def forward(self, x, sigma):
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) * (self.TARGET_STD / self.std.clamp(min=1e-12))
            
        if sigma is not None:
            noise = torch.randn_like(x)
            x = x + noise * sigma.view(-1, 1, 1, 1)
        else:
            sigma = torch.full((x.shape[0],), 1e-5, device=x.device)
            
        emb = self.encode_time(sigma)
        
        stage_indices = [1, 3, 5, 7]
        film_idx = 0
        
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in stage_indices:
                film_params = self.film_projs[film_idx](emb)
                scale, shift = film_params.chunk(2, dim=1)
                x = x * (1 + scale.view(*scale.shape, 1, 1)) + shift.view(*shift.shape, 1, 1)
                film_idx += 1
                
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier(x)
        return x

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

        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        latent_path = self.latent_paths[idx]
        
        latent = torch.load(latent_path + ".pt", weights_only=True)
            
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = self.img_transform(img)

        y = np.array(self.labels[idx])
        return latent.float(), img.float(), torch.from_numpy(y).long()

def train_distill_noise_cond(student, teacher, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, alpha):
    student.train()
    teacher.eval()
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    running_loss = 0.
    y_true, y_hat = [], []
    mse_loss_fn = torch.nn.MSELoss()
    for i, (latent, img, y) in pbar:
        latent, img, y = latent.to(device), img.to(device), y.to(device)
        
        sigma_min, sigma_max, rho = 0.002, 80, 7
        full_steps = 32
        step_indices = torch.randint(0, full_steps, (latent.shape[0],), device=device, dtype=torch.float32)
        sigma = (sigma_max ** (1/rho) + step_indices / (full_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        
        with torch.no_grad():
            teacher_logits = teacher(img)
            
        student_logits = student(latent, sigma)
        
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

def validate_distill_noise_cond(student, teacher, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, alpha):
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
            
            student_logits = student(latent, sigma=None)
            
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

def evaluate_distill_noise_cond(model, device, dataset, batch_size, model_dir, weights):
    model.load_state_dict(weights)
    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc='[TEST] EVALUATION')
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (latent, img, y) in pbar:
            latent = latent.to(device)
            out = model(latent, sigma=None)
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
    
    true_df = pd.DataFrame(y_true, columns=['label'])
    true_df.to_csv(os.path.join(model_dir, 'test_true.csv'), index=False)
    
    summary = f'Balanced Accuracy: {round(b_acc, 4)}\n'
    summary += f'Matthews Correlation Coefficient: {round(mcc, 4)}\n'
    summary += f'Mean AUC: {round(auc, 4)}\n\n'
    summary += 'Class:| Accuracy\n'
    for i, c in enumerate(dataset.CLASSES):
        summary += f'{c}:| {round(accuracies[i], 4)}\n'
    
    with open(os.path.join(model_dir, 'test_summary.txt'), 'w') as f:
        f.write(summary)

def save_setting_summary(args, model_dir, n_classes, classes):
    summary_path = os.path.join(model_dir, 'setting_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Training Setting Summary\n")
        f.write("========================\n\n")
        f.write(f"Model Directory: {model_dir}\n")
        f.write(f"Model Name: {args.model_name}\n")
        f.write(f"Number of Classes: {n_classes}\n")
        f.write(f"Classes: {', '.join(classes)}\n")
        
        f.write("Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")

def perform_crossfold(filelist: str, fold_num: int): 
    seed = 0 
    df = pd.read_csv(filelist, low_memory=False)
    metadata_columns = {"path", "id", "Split", "Unnamed: 0", "subject_id", "study_id", "dicom_id", "impression", "image", "Finding Labels", "FileName", "index"}
    class_columns = sorted([c for c in df.columns if c not in metadata_columns])
    
    labels = df[class_columns].idxmax(axis=1)
    groups = df['id'] if 'id' in df.columns else df.index
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df, labels, groups=groups))
    
    test_idx = folds[fold_num][1]
    val_idx = folds[(fold_num + 1) % 5][1]
    
    df['Split'] = 'TRAIN'
    df.loc[test_idx, 'Split'] = 'TEST'
    df.loc[val_idx, 'Split'] = 'VAL'
    
    df.to_csv(filelist, index=False)

def main(args):
    MODEL_NAME = 'cxr-lt'
    MODEL_NAME += f'_{args.model_name}_noisecond_distill'
    MODEL_NAME += f'_drw' if args.drw else ''
    MODEL_NAME += f'_alpha-{args.alpha}'
    lr_str = f"{args.lr:.4f}" if args.lr >= 1e-4 else f"{args.lr:g}"
    MODEL_NAME += f'_lr-{lr_str}_bs-{args.batch_size}'
    MODEL_NAME += f'_fold_{args.fold}' if args.do_crossfold else ''

    model_dir = os.path.join(args.out_dir, MODEL_NAME)
    
    if not args.test_only and os.path.exists(os.path.join(model_dir, 'test_summary.txt')):
        print(f"Skipping {MODEL_NAME} as test_summary.txt already exists.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.test_only and os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    set_seed(args.seed)

    filelist_to_use = args.filelist
    if args.do_crossfold:
        dataset_name = os.path.basename(args.filelist).split('.')[0]
        new_filelist = os.path.join(model_dir, f"{dataset_name}_fold_{args.fold}.csv")
        shutil.copy(args.filelist, new_filelist)
        filelist_to_use = new_filelist
        perform_crossfold(new_filelist, args.fold)

    std_path = args.mean_path.replace('_mean.pt', '_std.pt') if args.mean_path else None
    train_dataset = DistillationDataset(data_dir=args.data_dir, latent_dir=args.latent_dir, filelist=filelist_to_use, split='TRAIN', mean_path=args.mean_path, std_path=std_path)
    val_dataset = DistillationDataset(data_dir=args.data_dir, latent_dir=args.latent_dir, filelist=filelist_to_use, split='VAL', mean_path=args.mean_path, std_path=std_path)
    test_dataset = DistillationDataset(data_dir=args.data_dir, latent_dir=args.latent_dir, filelist=filelist_to_use, split='TEST', mean_path=args.mean_path, std_path=std_path)

    N_CLASSES = len(train_dataset.CLASSES)
    save_setting_summary(args, model_dir, N_CLASSES, train_dataset.CLASSES)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=val_worker_init_fn)

    history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'balanced_acc', 'mcc', 'auroc'])
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    device = torch.device('cuda:0')

    sample_latent, _, _ = train_dataset[0]
    in_channels, d, h = sample_latent.shape
    
    if args.model_name == 'ConvNeXt-Tiny':
        student_backbone = models.convnext_tiny(pretrained=(not args.rand_init))
        if in_channels != 3:
            original_conv = student_backbone.features[0][0]
            student_backbone.features[0][0] = nn.Conv2d(in_channels, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias is not None)
        
        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            state_dict = checkpoint.get('weights', checkpoint)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.2.')}
            student_backbone.load_state_dict(state_dict, strict=False)

        student_backbone.classifier[2] = nn.Linear(student_backbone.classifier[2].in_features, N_CLASSES)
    else:
        student_backbone = getattr(models, args.model_name)(pretrained=(not args.rand_init))
        if in_channels != 3:
            original_conv = student_backbone.conv1
            student_backbone.conv1 = nn.Conv2d(in_channels, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias is not None)
            
        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            state_dict = checkpoint.get('weights', checkpoint)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            student_backbone.load_state_dict(state_dict, strict=False)

        student_backbone.fc = nn.Linear(student_backbone.fc.in_features, N_CLASSES)

    mean, std = None, None
    if args.mean_path and os.path.exists(args.mean_path):
        mean = torch.load(args.mean_path, map_location='cpu', weights_only=True).view(1, -1, 1, 1).to(device)
        std_path = args.mean_path.replace('_mean.pt', '_std.pt')
        if os.path.exists(std_path):
            std = torch.load(std_path, map_location='cpu', weights_only=True).view(1, -1, 1, 1).to(device)
            
    student = NoiseConditionedConvNeXt(student_backbone, mean=mean, std=std).to(device)

    loss_fxn = get_loss(args, None, train_dataset)
    optimizer = torch.optim.Adam(student.backbone.classifier[2].parameters() if args.model_name == 'ConvNeXt-Tiny' else student.backbone.fc.parameters(), lr=args.lr)

    if args.teacher_model == 'ConvNeXt-Tiny':
        teacher = models.convnext_tiny(pretrained=True)
        if hasattr(teacher.classifier[2], "out_features") and teacher.classifier[2].out_features != N_CLASSES:
            teacher.classifier[2] = nn.Linear(teacher.classifier[2].in_features, N_CLASSES)
    else:
        teacher = getattr(models, args.teacher_model)(pretrained=True)
        if hasattr(teacher, "fc") and getattr(teacher.fc, "out_features", None) != N_CLASSES:
            teacher.fc = nn.Linear(teacher.fc.in_features, N_CLASSES)

    if args.teacher_weights_dir is not None:
        import glob
        if args.do_crossfold:
            search_pattern = os.path.join(args.teacher_weights_dir, f"*_fold_{args.fold}", "best.pt")
            matched_paths = glob.glob(search_pattern)
            teacher_path = matched_paths[0]
        else:
            teacher_path = args.teacher_weights_dir
            
        checkpoint = torch.load(teacher_path, map_location='cpu')
        state_dict = checkpoint.get('weights', checkpoint)
        teacher.load_state_dict(state_dict, strict=False)

    teacher = teacher.to(device)
    teacher.eval()

    if not args.test_only:
        epoch = 1
        early_stopping_dict = {'best_acc': 0., 'epochs_no_improve': 0}
        best_model_wts = None
        
        while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
            history = train_distill_noise_cond(student, teacher, device, loss_fxn, optimizer, train_loader, history, epoch, model_dir, args.alpha)
            history, early_stopping_dict, best_model_wts = validate_distill_noise_cond(student, teacher, device, loss_fxn, optimizer, val_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, args.alpha)
            
            if args.drw and epoch == args.drw_epoch:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
                loss_fxn = get_loss(args, None, train_dataset)
                early_stopping_dict['epochs_no_improve'] = 0

            epoch += 1
    else:
        best_model_wts = torch.load(os.path.join(model_dir, 'best.pt'), map_location='cpu')['weights']

    evaluate_distill_noise_cond(student, device, test_dataset, args.batch_size, model_dir, best_model_wts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/ssd1/greg/NIH_CXR/images', type=str)
    parser.add_argument('--latent_dir', required=True, type=str, help='directory containing latents')
    parser.add_argument('--filelist', required=True, type=str)
    parser.add_argument('--out_dir', default='results/', type=str)
    parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal', 'ldam'])
    parser.add_argument('--drw', action='store_true', default=False)
    parser.add_argument('--drw_epoch', default=10, type=int)
    parser.add_argument('--alpha', default=0.5, type=float, help='KD tradeoff; 1.0=CE only')
    parser.add_argument('--teacher_model', default='ConvNeXt-Tiny', type=str)
    parser.add_argument('--teacher_weights_dir', default=None, type=str)
    parser.add_argument('--model_name', default='ConvNeXt-Tiny', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--max_epochs', default=60, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--mean_path', type=str, default=None)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--do_crossfold', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)

    args = parser.parse_args()

    if args.do_crossfold:
        for fold in range(5): 
            args.seed = fold
            setattr(args, "fold", fold)
            main(args)
    else:
        main(args)
