"""
dataset.py — Data loading, transforms, and split utilities.
"""

import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]


train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps an ImageFolder subset with its own transform."""
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_loaders(pet_dir: str, batch_size: int = 32, seed: int = 42):
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    Split: 70% train / 15% val / 15% test.
    """
    random.seed(seed)
    full = datasets.ImageFolder(root=pet_dir)
    full.transform = None   # return raw PIL

    total      = len(full)
    train_size = int(0.70 * total)
    val_size   = int(0.15 * total)

    indices = list(range(total))
    random.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]

    train_ds = SubsetWithTransform(full, train_idx, train_transforms)
    val_ds   = SubsetWithTransform(full, val_idx,   eval_transforms)
    test_ds  = SubsetWithTransform(full, test_idx,  eval_transforms)

    kwargs = dict(num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, full.classes
