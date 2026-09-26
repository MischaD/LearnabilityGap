import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as T
from PIL import Image
from src.utils import hash_dataset_path


class ImageDataset(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        """
        Args:
            file_list (List[str]): Relative paths to images under root_dir.
            root_dir (str): Base directory for all image paths.
            transform (callable, optional): Transform to be applied on a PIL image producing a Tensor.
        """
        self.image_list = file_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        rel_path = self.image_list[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # Tensor expected
        return image, idx, rel_path


def get_data(config):
    data_csv = pd.read_csv(config.filelist)
    file_list = list(data_csv["path"])
    if hasattr(config, "debug") and config.debug: 
        file_list = file_list[:1000]
    output_filename = hash_dataset_path(os.path.dirname(config.filelist), "".join(file_list))
    return file_list, os.path.basename(output_filename)

def get_distributed_image_dataloader(file_list, rank, world_size, config, base_name=None):
    if base_name is None:
        base_name = os.path.dirname(config.filelist) if config.filelist.endswith(".csv") else config.filelist

    # Build transform: resize to expected model input size, to tensor in [0,1], then scale to [-1,1]
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = ImageDataset(file_list=file_list, root_dir=base_name, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        prefetch_factor=1,
        pin_memory=True,
    )
    return dataloader


