# io.py
from pathlib import Path
from typing import Optional, Tuple
import os
import numpy as np
import tifffile
import torch
from .perceptual import apply_lab_perturbation
from .utils import unnormalize, DEFAULT_MEAN, DEFAULT_STD
try:
    from skimage import color as skcolor 
except Exception:
    skcolor = None


def save_adversarial_tif(
    adv_img_tensor: torch.Tensor,
    index_in_batch: int,
    orig_path: str,
    labels_cpu: np.ndarray,
    out_dir: str,
    l_scale: float = 0.05,
    ab_scale: float = 0.4,
    smooth_sigma: float = 1.5,
):
    """
    Save adversarial image with minimal visual artifacts using LAB-based perturbation.
    
    Args:
        adv_img_tensor: (B,C,H,W) tensor with adversarial images in normalized [0,1] range.
        index_in_batch: index of the image in the batch to save.
        orig_path: path to original raw image (used to preserve true colors).
        labels_cpu: numpy array of true labels.
        out_dir: folder to save images.
        l_scale, ab_scale: LAB channel perturbation scaling.
        smooth_sigma: Gaussian smoothing sigma for LAB delta.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract adversarial image in [0,1]
    adv_img = adv_img_tensor[index_in_batch].detach().cpu().numpy().transpose(1,2,0)
    adv_img = np.clip(adv_img, 0.0, 1.0)
    
    # Load original raw image
    raw = load_and_prepare(orig_path, size=adv_img.shape[:2], is_raw=True)
    
    # Apply perceptual LAB perturbation
    adv_recon = apply_lab_perturbation(
        orig_rgb_raw=raw,
        adv_rgb_scaled_01=adv_img,
        l_scale=l_scale,
        ab_scale=ab_scale,
        smooth_sigma=smooth_sigma
    )
    
    # Convert to uint8 [0,255]
    img_final = np.clip(adv_recon, 0, 1)  # Ensure no overshoot
    img_final = (img_final * 255).astype(np.uint8)
    
    # Build filename
    base = Path(orig_path).stem
    true_label = int(labels_cpu[index_in_batch])
    fname_tif = f"{base}_true{true_label}.tif"
    out_path = Path(out_dir) / fname_tif
    
    # Save
    tifffile.imwrite(str(out_path), img_final, photometric='rgb')
    return str(out_path)


def load_tif_image(path: str, normalize: bool = True) -> np.ndarray:
    """
    Load a multi-band .tif image as a numpy array (C,H,W), optionally normalized to [0,1].

    Args:
        path: Path to the .tif file.
        normalize: If True, scale values to [0,1] based on data range.

    Returns:
        np.ndarray: Image array in (C,H,W) format, dtype=float32 if normalized.
    """
    import tifffile
    img = tifffile.imread(path)  # Usually (H, W, C)
    if img.ndim == 2:
        img = img[..., np.newaxis]  # (H, W, 1)

    # Convert to (C,H,W)
    img = np.transpose(img, (2, 0, 1))

    if normalize:
        img = img.astype(np.float32)
        vmin, vmax = np.percentile(img, (1, 99.9))
        if vmax > vmin:
            img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        else:
            img = np.zeros_like(img, dtype=np.float32)

    return img
