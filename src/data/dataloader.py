import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Rasterio to read TIFF files
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.endswith(".tif"):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        with rasterio.open(img_path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32) / 255.0

        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        return img, label


def compute_mean_std(dataset, sample_size=None):
    """
    Calculate mean and std of EuroSATDataset
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    n_samples = len(dataset) if sample_size is None else min(sample_size, len(dataset))

    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        if total_images >= n_samples:
            break
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std


def get_dataloaders(data_dir="data/raw", batch_size=32, val_split=0.15, test_split=0.15, seed=42):
    """
    DataLoader to load EuroSAT data
    """
    base_dataset = EuroSATDataset(data_dir, transform=transforms.ToTensor())
    mean, std = compute_mean_std(base_dataset, sample_size=2000)
    std[std == 0] = 1.0
    
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    full_dataset = EuroSATDataset(data_dir, transform=train_transform)
    class_names = full_dataset.class_names

    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names
