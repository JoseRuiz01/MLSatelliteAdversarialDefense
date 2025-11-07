# utils.py
from typing import Tuple, Optional
from dataclasses import dataclass
import torch
import numpy as np

DEFAULT_MEAN = [0.3443, 0.3803, 0.4082]
DEFAULT_STD = [0.1573, 0.1309, 0.1198]


def extract_mean_std(dataloader) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Attempt to extract Normalize(mean,std) from dataloader.dataset.transform.
    Returns (mean_tensor, std_tensor) in shape (C,1,1) or (None, None).
    """
    try:
        ds = dataloader.dataset
        if hasattr(ds, "dataset"):
            base = ds.dataset
        else:
            base = ds
        transform = getattr(base, "transform", None)
        if transform and hasattr(transform, "transforms"):
            for t in transform.transforms:
                if t.__class__.__name__ == "Normalize":
                    mean = torch.tensor(t.mean).view(-1, 1, 1)
                    std = torch.tensor(t.std).view(-1, 1, 1)
                    return mean, std
    except Exception:
        pass
    return None, None


def unnormalize(img: torch.Tensor, mean, std) -> torch.Tensor:
    """
    img: (C,H,W) or (B,C,H,W). mean/std can be tensor or list.
    returns image in same device as input.
    """
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean).view(-1, 1, 1).to(img.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std).view(-1, 1, 1).to(img.device)
    return img * std + mean


def select_rgb_bands(img):
    """Selects RGB bands matching GDAL example: bands 4, 3, 2."""
    if img.ndim == 3 and img.shape[2] >= 4:
        # Using 0-based indexing: Band 4→3, Band 3→2, Band 2→1
        return img[:, :, [3, 2, 1]]
    elif img.ndim == 2:
        return np.stack([img]*3, axis=-1)
    else:
        return img[:, :, :3]


def gdal_style_scale(img, src_min=0, src_max=2750, dst_min=1, dst_max=255):
    """Apply GDAL-style scaling and clipping."""
    img = img.astype(np.float32)
    img = np.clip(img, src_min, src_max)
    img = (img - src_min) / (src_max - src_min)
    img = img * (dst_max - dst_min) + dst_min
    img = np.clip(img, dst_min, dst_max)
    return (img / 255.0).astype(np.float32)  # normalize for display
