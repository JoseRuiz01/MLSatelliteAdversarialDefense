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
    grad_blur_sigma: float = 2.0
    smooth_perturb_sigma: float = 1.0
    random_dither: bool = True
    dither_scale: float = 0.5
    device: Optional[torch.device] = None


def build_importance_mask(
    grad: torch.Tensor,
    keep_fraction: float = 0.3,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    
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

import torch.nn.functional as F

def apply_low_pass_filter(perturbation: torch.Tensor, cutoff_ratio: float = 0.8) -> torch.Tensor:
    """Apply frequency domain low-pass filter to perturbation."""
    B, C, H, W = perturbation.shape
    filtered = torch.zeros_like(perturbation)
    
    for b in range(B):
        for c in range(C):
            # FFT
            freq = torch.fft.fft2(perturbation[b, c])
            freq_shifted = torch.fft.fftshift(freq)
            
            # Create circular mask
            crow, ccol = H // 2, W // 2
            radius = int(min(H, W) * cutoff_ratio / 2)
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            mask = ((x - ccol)**2 + (y - crow)**2) <= radius**2
            mask = mask.to(perturbation.device)
            
            # Apply mask and inverse FFT
            freq_shifted = freq_shifted * mask
            freq = torch.fft.ifftshift(freq_shifted)
            filtered[b, c] = torch.fft.ifft2(freq).real
    
    return filtered


def smooth_grad_torch(grad: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Smooth gradient maps by convolving with a Gaussian-like separable kernel.
    grad: (B,C,H,W)
    """
    if sigma <= 0 or kernel_size <= 1:
        return grad

    # build 1D gaussian kernel
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=grad.device)
    kernel1d = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    kernel1d = kernel1d / kernel1d.sum()
    # separable via outer-product -> 2D kernel
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # (1,1,kH,kW)

    C = grad.shape[1]
    # replicate kernel for group conv
    kernel = kernel2d.repeat(C, 1, 1, 1)  # (C,1,kH,kW)
    # pad and convolve (grouped conv)
    padding = kernel_size // 2
    grad_sm = F.conv2d(grad, kernel, groups=C, padding=padding)
    return grad_sm


def compute_gradient_step(
    model: nn.Module,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool,
    target_labels: Optional[torch.Tensor],
    loss_fn=nn.CrossEntropyLoss(),
) -> torch.Tensor:
   
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

    if gaussian_filter is None or sigma <= 0:
        return pert
    out = pert.copy()
    B, C, H, W = pert.shape
    for b in range(B):
        for c in range(C):
            out[b, c] = gaussian_filter(pert[b, c], sigma=sigma)
    return out

def compute_saliency_mask(images: torch.Tensor) -> torch.Tensor:
    """Compute edge-based saliency to avoid perturbing edges."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    
    gray = images.mean(dim=1, keepdim=True)
    edges_x = F.conv2d(gray, sobel_x, padding=1)
    edges_y = F.conv2d(gray, sobel_y, padding=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    
    mask = 1.0 - torch.sigmoid(edges * 5)
    return mask.repeat(1, images.shape[1], 1, 1)


def pgd_attack_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: PGDConfig,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PGD attack que funciona correctamente en espacio normalizado.
    """
    device = config.device or images.device
    images = images.to(device)
    labels = labels.to(device)
    
    ori_images = images.clone().detach()
    adv_images = ori_images.clone().detach()
    
    eps = config.eps
    alpha = config.alpha if config.alpha is not None else eps / 10
    
    loss_fn = nn.CrossEntropyLoss()
    
    for iteration in range(config.iters):
        adv_images.requires_grad_(True)
        
        outputs = model(adv_images)
        
        def compute_perceptual_loss(original: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
            """Compute perceptual difference using feature extraction."""
            return F.mse_loss(original, adversarial)
        
        if targeted and target_labels is not None:
            attack_loss = -loss_fn(outputs, target_labels.to(device))
        else:
            attack_loss = loss_fn(outputs, labels)

        perceptual_penalty = compute_perceptual_loss(ori_images, adv_images)
        loss = attack_loss + 0.1 * perceptual_penalty  
      
        model.zero_grad()
        loss.backward()
        grad = adv_images.grad.data  # (B,C,H,W)

        if config.grad_blur_sigma and config.grad_blur_sigma > 0:
            ks = max(3, int(6 * config.grad_blur_sigma) | 1)
            grad = smooth_grad_torch(grad, kernel_size=ks, sigma=config.grad_blur_sigma)

        with torch.no_grad():
            adv_images = adv_images + alpha * grad.sign()
            
            perturbation = adv_images - ori_images
            perturbation = apply_low_pass_filter(perturbation, cutoff_ratio=0.7)
            saliency_mask = compute_saliency_mask(ori_images)
            perturbation = perturbation * saliency_mask  
            perturbation = torch.clamp(perturbation, -eps, eps)
            adv_images = ori_images + perturbation
    
    return adv_images.detach()