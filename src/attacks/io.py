# io.py
from pathlib import Path
from typing import Optional, Tuple
import os
import numpy as np
import tifffile

from .perceptual import apply_lab_perturbation
from .utils import unnormalize, DEFAULT_MEAN, DEFAULT_STD
try:
    from skimage import color as skcolor 
except Exception:
    skcolor = None


def save_adversarial_tif(
    adv_img_tensor,        # (C,H,W) or (B,C,H,W) torch.Tensor in normalized space
    index_in_batch: int,
    orig_path: str,
    mean_t,
    std_t,
    labels_cpu,
    preds_cpu,
    out_dir: str,
    eps: float,
    perceptual_eps_factor: float = 0.45,
    smooth_sigma: float = 1.0,
    dither_scale: float = 0.3,
):
    """
    Save one adversarial TIFF modifying only raw RGB bands.
    This mirrors the logic from the original script but is encapsulated here.

    Args:
        adv_img_tensor: tensor on CPU or device (C,H,W) or (B,C,H,W)
        index_in_batch: which sample index when adv_img_tensor is batched
        orig_path: path to original raw tif
        mean_t, std_t: normalization tensors shape (C,1,1)
        labels_cpu, preds_cpu: arrays or tensors with labels & preds (for filename)
        out_dir: directory to save to
        eps: epsilon in normalized space (relative)
        perceptual_eps_factor: scale applied when mapping eps -> raw units
    """
    os.makedirs(out_dir, exist_ok=True)
    if hasattr(adv_img_tensor, "ndim") and adv_img_tensor.ndim == 4:
        img = adv_img_tensor[index_in_batch]
    else:
        img = adv_img_tensor

    import torch
    device = img.device if torch.is_tensor(img) else None
    img_unn = unnormalize(img.to(device), mean_t.to(device), std_t.to(device)).cpu().numpy()
    # (C,H,W) -> H,W,C
    if img_unn.ndim == 3:
        img_unn = np.transpose(img_unn, (1, 2, 0))

    raw_img = tifffile.imread(orig_path)
    raw_dtype = raw_img.dtype
    raw_min, raw_max = np.min(raw_img), np.max(raw_img)
    if raw_max == raw_min:
        raw_max = raw_min + 1.0

    img_scaled = np.clip(img_unn, 0.0, 1.0)
    img_scaled = img_scaled * (raw_max - raw_min) + raw_min
    img_scaled = np.clip(img_scaled, raw_min, raw_max)

    rgb_raw_indices = [3, 2, 1]  

    img_final = raw_img.copy().astype(np.int64)
    eps_raw = max(1, int(round(eps * (raw_max - raw_min) * perceptual_eps_factor)))

    orig_rgb = img_final[..., rgb_raw_indices].astype(np.float32)
    adv_rgb_01 = (img_scaled - raw_min) / (raw_max - raw_min)
    adv_rgb_01 = np.clip(adv_rgb_01, 0.0, 1.0)

    try:
        rgb_recons_01 = apply_lab_perturbation(
            orig_rgb_raw=orig_rgb,
            adv_rgb_scaled_01=adv_rgb_01,
            raw_min=raw_min,
            raw_max=raw_max,
            smooth_sigma=smooth_sigma,
        )
    except Exception:
        rgb_recons_01 = adv_rgb_01

    rgb_recons_raw = np.clip(np.round(rgb_recons_01 * (raw_max - raw_min) + raw_min), raw_min, raw_max).astype(raw_dtype)

    for k, raw_idx in enumerate(rgb_raw_indices):
        orig_band = img_final[..., raw_idx].astype(np.int64)
        adv_band = rgb_recons_raw[..., k].astype(np.int64)
        diff = (adv_band - orig_band).astype(np.int64)
        diff = np.clip(diff, -eps_raw, eps_raw)
        if dither_scale > 0:
            noise = (np.random.rand(*diff.shape) - 0.5) * 2 * max(1, int(dither_scale * eps_raw))
            diff = diff + noise.astype(np.int64)
        img_final[..., raw_idx] = np.clip(orig_band + diff, raw_min, raw_max)

    img_final = np.clip(img_final, raw_min, raw_max).astype(raw_dtype)

    base = Path(orig_path).stem
    fname_tif = f"{base}_true{int(labels_cpu[index_in_batch])}_pred{int(preds_cpu[index_in_batch])}.tif"
    out_path = Path(out_dir) / fname_tif
    tifffile.imwrite(str(out_path), img_final)
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
