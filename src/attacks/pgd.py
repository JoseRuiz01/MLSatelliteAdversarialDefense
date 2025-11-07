# pgd.py
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# optional imports
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


@dataclass
class PGDConfig:
    eps: float = 0.005
    alpha: Optional[float] = None   
    iters: int = 50
    small_step_fraction: float = 0.2
    grad_mask_fraction: float = 0.25
    grad_blur_sigma: float = 1.0
    smooth_perturb_sigma: float = 1.0
    random_dither: bool = True
    dither_scale: float = 0.5
    device: Optional[torch.device] = None


def build_importance_mask(
    grad: torch.Tensor,
    keep_fraction: float = 0.3,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Build a soft per-pixel importance mask from gradients.

    Args:
        grad: gradient tensor of shape (B, C, H, W)
        keep_fraction: fraction of pixels to keep (highest magnitude)
        blur_sigma: gaussian blur sigma applied to binary mask (optional)
    """
    with torch.no_grad():
        magnitude = grad.abs().mean(dim=1)  # (B, H, W)
        B, H, W = magnitude.shape
        mask = torch.zeros_like(magnitude)

        k = max(1, int(H * W * keep_fraction))

        for i in range(B):
            flat = magnitude[i].view(-1)
            if k >= flat.numel():
                top_mask = torch.ones_like(flat)
            else:
                kth_idx = flat.numel() - k + 1
                thresh = torch.kthvalue(flat, kth_idx).values
                top_mask = (flat >= thresh).float()
            top_mask = top_mask.view(H, W).cpu().numpy()
            if gaussian_filter is not None and blur_sigma > 0:
                top_mask = gaussian_filter(top_mask, sigma=blur_sigma)
            else:
                kernel = np.ones((3, 3)) / 9.0
                top_mask = np.clip(
                    np.convolve(top_mask.ravel(), kernel.ravel(), mode="same"), 0, 1
                ).reshape(H, W)

            if top_mask.max() > 0:
                top_mask = top_mask / top_mask.max()

            mask[i] = torch.from_numpy(top_mask)

        mask = mask.unsqueeze(1).to(grad.device)  # (B,1,H,W)
        return mask


def compute_gradient_step(
    model: nn.Module,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool,
    target_labels: Optional[torch.Tensor],
    loss_fn=nn.CrossEntropyLoss(),
) -> torch.Tensor:
    """
    Compute gradient with respect to inputs (no update) and return step direction.
    If targeted=True the step is negative gradient (move toward target).
    """
    imgs = imgs.clone().detach().requires_grad_(True)
    outputs = model(imgs)
    if targeted:
        if target_labels is None:
            raise ValueError("target_labels must be provided when targeted=True")
        loss = loss_fn(outputs, target_labels)
        grad = torch.autograd.grad(loss, imgs)[0]
        return -grad
    else:
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, imgs)[0]
        return grad


def _apply_gaussian_smooth_to_perturbation(pert: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply gaussian smoothing to perturbation stored as numpy (B,C,H,W).
    Fallback if scipy gaussian_filter is not available.
    """
    if gaussian_filter is None or sigma <= 0:
        return pert
    out = pert.copy()
    B, C, H, W = pert.shape
    for b in range(B):
        for c in range(C):
            out[b, c] = gaussian_filter(pert[b, c], sigma=sigma)
    return out


def pgd_attack_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: PGDConfig,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PGD with importance masking, smoothing and optional dithering.

    Args:
        model: model to attack
        images: (B,C,H,W) normalized images (tensor)
        labels: (B,) ground-truth labels (tensor)
        config: PGDConfig instance
        targeted: whether attack is targeted
        target_labels: (B,) target labels if targeted

    Returns:
        adv_images: adversarial images tensor (B,C,H,W)
    """
    device = config.device or images.device
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    labels = labels.to(device)
    adv_images = images.clone().detach()

    eps = config.eps
    alpha = config.alpha if config.alpha is not None else eps * config.small_step_fraction

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(config.iters):
        adv_images.requires_grad_(True)

        outputs = model(adv_images)
        if targeted:
            if target_labels is None:
                raise ValueError("target_labels must be provided for targeted attack")
            loss = loss_fn(outputs, target_labels.to(device))
            grad = torch.autograd.grad(loss, adv_images)[0]
            step = -grad
        else:
            loss = loss_fn(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]
            step = grad

        # importance mask
        mask = build_importance_mask(grad, keep_fraction=config.grad_mask_fraction, blur_sigma=config.grad_blur_sigma)
        step = step * mask

        # normalize step per-sample to avoid variable magnitudes
        step_mean = step.abs().mean(dim=(1, 2, 3), keepdim=True)
        step_norm = step / (step_mean + 1e-10)

        # take step
        adv_images = adv_images + alpha * step_norm
        delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + delta, min=ori_images.min().item() - 1.0, max=ori_images.max().item() + 1.0).detach()

        # smoothing of perturbation in pixel-space (optional)
        if config.smooth_perturb_sigma > 0:
            with torch.no_grad():
                pert = (adv_images - ori_images).cpu().numpy()  # B,C,H,W
                pert = _apply_gaussian_smooth_to_perturbation(pert, sigma=config.smooth_perturb_sigma)
                adv_images = torch.from_numpy(pert).to(device).float() + ori_images
                delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
                adv_images = torch.clamp(ori_images + delta, min=ori_images.min().item() - 1.0, max=ori_images.max().item() + 1.0).detach()

        # optional dithering
        if config.random_dither:
            with torch.no_grad():
                noise = (torch.rand_like(adv_images) - 0.5) * (config.dither_scale * eps)
                adv_images = torch.clamp(adv_images + noise, min=ori_images.min().item() - 1.0, max=ori_images.max().item() + 1.0).detach()

    return adv_images.detach()
