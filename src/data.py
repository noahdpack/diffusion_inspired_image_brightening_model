import os
import random
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def to_minus_one_to_one(t: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] tensor to [-1, 1]."""
    return (t * 2) - 1


class PairedTransform:
    """
    Apply the same random spatial + pixel transforms to a pair of images.

    This is used to keep (dark, bright) pairs aligned while augmenting.
    Images are converted to tensors in [-1, 1].
    """

    def __init__(self, img_size: int) -> None:
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(to_minus_one_to_one),  # [0,1] -> [-1,1]
        ])

    def __call__(self, img1: Image.Image, img2: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use the same random seed for both images so augmentations match
        seed = np.random.randint(2147483647)

        random.seed(seed)
        torch.manual_seed(seed)
        img1 = self.transform(img1)

        random.seed(seed)
        torch.manual_seed(seed)
        img2 = self.transform(img2)

        return img1, img2


class PairedImageDataset(Dataset):
    """
    Dataset of paired images stored in two folders:
    - dark_dir: directory with darker versions
    - bright_dir: directory with brighter/ground truth versions

    Assumes matching filenames in both directories.
    """

    def __init__(
        self,
        dark_dir: str,
        bright_dir: str,
        transform: Optional[PairedTransform] = None,
    ) -> None:
        self.dark_dir = dark_dir
        self.bright_dir = bright_dir
        self.transform = transform

        # Assume matching filenames in both directories
        self.filenames = sorted(os.listdir(dark_dir))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dark_path = os.path.join(self.dark_dir, self.filenames[idx])
        bright_path = os.path.join(self.bright_dir, self.filenames[idx])

        dark_img = Image.open(dark_path).convert("RGB")
        bright_img = Image.open(bright_path).convert("RGB")

        if self.transform is not None:
            dark_img, bright_img = self.transform(dark_img, bright_img)

        return dark_img, bright_img


def build_dataloader(
    dark_dir: str,
    bright_dir: str,
    img_size: int = 64,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """
    Convenience helper to build a DataLoader with the PairedImageDataset.
    Note: although the dataset returns (dark, bright), the current experiment
    only uses the bright image as the ground truth x0.
    """
    paired_transform = PairedTransform(img_size=img_size)

    dataset = PairedImageDataset(
        dark_dir=dark_dir,
        bright_dir=bright_dir,
        transform=paired_transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return dataloader
