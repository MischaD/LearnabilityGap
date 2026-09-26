import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

class BaseClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, filelist, split):
        self.data_dir = data_dir
        self.split = split
        self.label_df = pd.read_csv(filelist)

        if self.split:
            self.label_df = self.label_df[self.label_df['Split'] == self.split]

        metadata_columns = {"path", "id", "Split", "Unnamed: 0", "subject_id", "study_id", "dicom_id", "impression", "image", "Finding Labels", "FileName", "index"}
        self.CLASSES = sorted([c for c in self.label_df.columns if c not in metadata_columns])

        if 'path' in self.label_df.columns:
            self.path_col = 'path'
        elif 'id' in self.label_df.columns:
            self.path_col = 'id'
        else:
            raise KeyError("Neither 'path' nor 'id' column found in CSV.")

        self.rel_paths = self.label_df[self.path_col].values.tolist()
        self.img_paths = [os.path.join(self.data_dir, p) for p in self.rel_paths]
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values
        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

    def __len__(self):
        return len(self.img_paths)

class ImageClassificationDataset(BaseClassificationDataset):
    def __init__(self, data_dir, filelist, split):
        super().__init__(data_dir, filelist, split)
        if self.split == 'TRAIN' or self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        x = cv2.imread(path)
        x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
        x = self.transform(x)
        y = np.array(self.labels[idx])
        return x.float(), torch.from_numpy(y).long()

class VAEImageDataset(BaseClassificationDataset):
    def __init__(self, data_dir, filelist, split):
        super().__init__(data_dir, filelist, split)
        if self.split == 'TRAIN' or self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        x = cv2.imread(path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = self.transform(x)
        y = np.array(self.labels[idx])
        return x.float(), torch.from_numpy(y).long()

class LatentClassificationDataset(BaseClassificationDataset):
    def __init__(self, data_dir, filelist, split, mask_ratio=0.0, return_rel_path=False, mean_path=None, std_path=None, mask_mode="default"):
        super().__init__(data_dir, filelist, split)
        self.mask_ratio = mask_ratio
        self.return_rel_path = return_rel_path
        self.mask_mode = mask_mode
        
        self.mean = None
        self.std = None
        if mean_path and os.path.exists(mean_path):
            self.mean = torch.load(mean_path, map_location='cpu', weights_only=True).view(-1, 1, 1)
        if std_path and os.path.exists(std_path):
            self.std = torch.load(std_path, map_location='cpu', weights_only=True).view(-1, 1, 1)
        
        self.TARGET_STD = 0.5

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        x = torch.load(path + ".pt", weights_only=True)
        
        if self.mask_ratio > 0 and (self.split is not None) and (self.split.upper().startswith('TR')):
            if self.mask_mode == "default": 
                mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device) > self.mask_ratio
                x = x * mask
            elif self.mask_mode == "cw": 
                mask = torch.rand(x.shape[0], 1, 1) > self.mask_ratio
                x = x * mask

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) * (self.TARGET_STD / self.std.clamp(min=1e-12))

        y = np.array(self.labels[idx])
        if self.return_rel_path:
            return x.float(), torch.from_numpy(y).long(), self.rel_paths[idx]
        return x.float(), torch.from_numpy(y).long()

def ClassificationDataset(data_dir, filelist, split, is_latent=False, mask_ratio=0.0, return_rel_path=False, mean_path=None, std_path=None, vae_mode=False, mask_mode="default"):
    if is_latent:
        return LatentClassificationDataset(data_dir, filelist, split, mask_ratio=mask_ratio, return_rel_path=return_rel_path, mean_path=mean_path, std_path=std_path, mask_mode=mask_mode)
    elif vae_mode:
        return VAEImageDataset(data_dir, filelist, split)
    else:
        return ImageClassificationDataset(data_dir, filelist, split)

## CREDIT TO https://github.com/agaldran/balanced_mixup ##

# pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches