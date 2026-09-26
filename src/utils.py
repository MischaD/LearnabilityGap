import os
import torch
import torchvision.transforms as transforms
import torchvision.io as io
from diffusers import AutoencoderKL
from einops import repeat
import hashlib


def hash_dataset_path(dataset_root_dir, img_list):
    """Takes a list of paths and joins it to a large string - then uses it as hash input stringuses it filename for the entire datsets for quicker loading"""
    name = "".join([x for x in img_list])
    name = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return os.path.join(dataset_root_dir, "hashdata_" + name)
