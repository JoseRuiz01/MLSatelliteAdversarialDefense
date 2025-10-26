import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import tifffile
from pathlib import Path
from skimage import color
from scipy.ndimage import gaussian_filter
try:
    from skimage import color
except Exception:
    color = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


DEFAULT_MEAN = [0.3443, 0.3803, 0.4082]
DEFAULT_STD  = [0.1573, 0.1309, 0.1198]


def extract_mean_std(dataloader):
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
                    mean = torch.tensor(t.mean).view(-1,1,1)
                    std  = torch.tensor(t.std).view(-1,1,1)
                    return mean, std
    except Exception:
        pass
    return None, None


def unnormalize(img, mean, std):
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean).view(-1,1,1).to(img.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std).view(-1,1,1).to(img.device)
    return img * std + mean


def _make_importance_mask(grad, keep_fraction=0.3, blur_sigma=1.0):
    """Builds a soft mask from gradient magnitude selecting `keep_fraction` of pixels.
    Returns mask with same HxW shape normalized to [0,1]."""
    # grad: (B,C,H,W)
    with torch.no_grad():
        magnitude = grad.abs().mean(dim=1, keepdim=False)  # (B,H,W)
        B, H, W = magnitude.shape
        mask = torch.zeros_like(magnitude)
        k = max(1, int(H*W * keep_fraction))
        # For each image create a soft mask where top-k pixels keep weight 1 and then gaussian blur
        for i in range(B):
            flat = magnitude[i].view(-1)
            if k >= flat.numel():
                top_mask = torch.ones_like(flat)
            else:
                thresh = torch.kthvalue(flat, flat.numel()-k+1).values
                top_mask = (flat >= thresh).float()
            top_mask = top_mask.view(H, W).cpu().numpy()
            if gaussian_filter is not None and blur_sigma > 0:
                top_mask = gaussian_filter(top_mask, sigma=blur_sigma)
            else:
                # simple 3x3 blur fallback
                kernel = np.ones((3,3))/9.0
                top_mask = np.clip(np.convolve(top_mask.ravel(), kernel.ravel(), mode='same'), 0, 1).reshape(H,W)
            # normalize per-image
            if top_mask.max() > 0:
                top_mask = top_mask / top_mask.max()
            mask[i] = torch.from_numpy(top_mask)
        return mask.unsqueeze(1).to(grad.device)  # (B,1,H,W)


def pgd_attack_batch(model, images, labels, eps, alpha, iters, device,
                     targeted=False, target_labels=None,
                     small_step_fraction=0.2,  # alpha = small_step_fraction * eps if alpha is None
                     grad_mask_fraction=0.25,
                     grad_blur_sigma=1.0,
                     smooth_perturb_sigma=1.0,
                     random_dither=True,
                     dither_scale=0.5):
    """
    Improved PGD: uses small steps, gradient-based spatial masking (so we only perturb important regions),
    and optional gaussian smoothing + tiny dithering so changes are less visually obvious.

    The attack operates in the model's normalized input space (same as original code). Only at save time
    we map back to raw and then clip per-band with an additional perceptual factor.
    """
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    labels = labels.to(device)
    adv_images = images.clone().detach()

    if alpha is None:
        alpha = eps * small_step_fraction

    loss_fn = nn.CrossEntropyLoss()

    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        if targeted:
            if target_labels is None:
                raise ValueError("target_labels must be provided for targeted=True")
            loss = loss_fn(outputs, target_labels.to(device))
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            step = -grad  # targeted: move toward target -> decrease loss
        else:
            loss = loss_fn(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            step = grad

        # Build importance mask from gradient magnitude (per-iteration, per-batch)
        mask = _make_importance_mask(grad, keep_fraction=grad_mask_fraction, blur_sigma=grad_blur_sigma)
        # mask: (B,1,H,W) values in [0,1] where 1 is most important

        # apply mask to gradient; normalize per-sample
        step = step * mask
        # l_inf style step but scaled by gradient magnitude (not sign) to reduce harsh edges
        step_norm = step / (step.abs().mean(dim=(1,2,3), keepdim=True) + 1e-10)
        adv_images = adv_images + alpha * step_norm

        # Project back into eps-ball around original (linf)
        delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + delta, min=ori_images.min().item()-1.0, max=ori_images.max().item()+1.0).detach()

        # Optional: apply small gaussian smoothing to the perturbation in image space to remove high-frequency artifacts
        if smooth_perturb_sigma > 0 and gaussian_filter is not None:
            with torch.no_grad():
                pert = (adv_images - ori_images).cpu().numpy()  # B,C,H,W
                for b in range(pert.shape[0]):
                    for c in range(pert.shape[1]):
                        pert[b,c] = gaussian_filter(pert[b,c], sigma=smooth_perturb_sigma)
                adv_images = torch.from_numpy(pert).to(device).float() + ori_images
                # re-project
                delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
                adv_images = torch.clamp(ori_images + delta, min=ori_images.min().item()-1.0, max=ori_images.max().item()+1.0).detach()

        # Optional small random dither to break visible patterning
        if random_dither:
            with torch.no_grad():
                noise = (torch.rand_like(adv_images) - 0.5) * (dither_scale * eps)
                adv_images = torch.clamp(adv_images + noise, min=ori_images.min().item()-1.0, max=ori_images.max().item()+1.0).detach()

    return adv_images.detach()


def _smooth_numpy(arr, sigma=1.0):
    if gaussian_filter is not None:
        return gaussian_filter(arr, sigma=sigma)
    else:
        # fallback: simple gaussian-like box blur using convolution (cheap)
        from scipy.signal import convolve2d
        kernel = np.ones((3,3)) / 9.0
        out = np.zeros_like(arr)
        if arr.ndim == 2:
            out = convolve2d(arr, kernel, mode='same', boundary='symm')
        else:
            for c in range(arr.shape[2]):
                out[..., c] = convolve2d(arr[..., c], kernel, mode='same', boundary='symm')
        return out


def evaluate_pgd(model, dataloader, device, eps=0.005, alpha=None, iters=50,
                 out_dir="../data/adversarial/pgd", save_every=20, max_save=200,
                 targeted=False, target_class=None,
                 # visibility-reduction params
                 perceptual_eps_factor=0.45,  # reduce raw eps further when mapping to raw
                 smooth_sigma=1.0,
                 grad_mask_fraction=0.25,
                 dither_scale=0.3):
    """
    Evaluate PGD and save adversarial .tif files modifying ONLY the raw RGB bands.
    Reduces visible artifacts via:
      - Smaller-step PGD with gradient masking
      - Gaussian smoothing of the perturbation
      - Random dithering
      - LAB-based perceptual scaling of RGB changes
    """
    from skimage import color
    from scipy.ndimage import gaussian_filter

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mean_t, std_t = extract_mean_std(dataloader)
    if mean_t is None or std_t is None:
        mean_t = torch.tensor(DEFAULT_MEAN).view(-1,1,1)
        std_t  = torch.tensor(DEFAULT_STD).view(-1,1,1)

    if alpha is None:
        alpha = eps * 0.2

    criterion = nn.CrossEntropyLoss()
    total = 0
    clean_correct = 0
    adv_correct = 0
    clean_loss_total = 0.0
    adv_loss_total = 0.0
    saved = 0
    batch_idx = 0
    global_ptr = 0

    for images, labels in tqdm(dataloader, desc=f"PGD eps={eps}", leave=False):
        batch_idx += 1
        images = images.to(device)
        labels = labels.to(device)
        total += images.size(0)

        with torch.no_grad():
            out = model(images)
            loss = criterion(out, labels)
            _, preds = out.max(1)
            clean_correct += (preds == labels).sum().item()
            clean_loss_total += loss.item() * images.size(0)

        target_labels = None
        if targeted:
            if target_class is None:
                with torch.no_grad():
                    probs = torch.softmax(out, dim=1)
                    target_labels = probs.argsort(dim=1)[:, -2]
            else:
                target_labels = torch.full_like(labels, fill_value=int(target_class), device=device)

        adv_images = pgd_attack_batch(
            model, images, labels, eps=eps, alpha=alpha, iters=iters,
            device=device, targeted=targeted, target_labels=target_labels,
            small_step_fraction=0.2, grad_mask_fraction=grad_mask_fraction,
            grad_blur_sigma=1.0, smooth_perturb_sigma=smooth_sigma,
            random_dither=True, dither_scale=dither_scale
        )

        with torch.no_grad():
            out_adv = model(adv_images)
            loss_adv = criterion(out_adv, labels)
            _, preds_adv = out_adv.max(1)
            adv_correct += (preds_adv == labels).sum().item()
            adv_loss_total += loss_adv.item() * images.size(0)

        # --- Save adversarial images ---
        if (batch_idx % save_every == 0) and (max_save is None or saved < max_save):
            adv_cpu = adv_images.cpu()
            labels_cpu = labels.cpu()
            preds_cpu = preds_adv.cpu()
            batch_size_cur = adv_cpu.size(0)

            for i in range(batch_size_cur):
                if max_save is not None and saved >= max_save:
                    break

                if hasattr(dataloader.dataset, "indices"):
                    idx = dataloader.dataset.indices[global_ptr + i]
                    orig_dataset = dataloader.dataset.dataset
                else:
                    idx = global_ptr + i
                    orig_dataset = dataloader.dataset

                orig_name = orig_dataset.samples[idx][0]
                base = os.path.splitext(os.path.basename(orig_name))[0]

                # Unnormalize adversarial to [0,1]
                img = adv_cpu[i]
                img_unn = unnormalize(img.to(device), mean_t.to(device), std_t.to(device)).cpu().numpy()
                if img_unn.ndim == 3:
                    img_unn = np.transpose(img_unn, (1,2,0))  # H,W,C

                # Load original raw image
                raw_img = tifffile.imread(orig_name)
                raw_dtype = raw_img.dtype
                raw_min, raw_max = np.min(raw_img), np.max(raw_img)

                # Clip adversarial to [0,1] and scale to raw
                img_scaled = np.clip(img_unn, 0.0, 1.0)
                img_scaled = img_scaled * (raw_max - raw_min) + raw_min
                img_scaled = np.clip(img_scaled, raw_min, raw_max)

                # RGB band indices
                rgb_raw_indices = [3, 2, 1]  # Sentinel-2 4,3,2

                img_final = raw_img.copy().astype(np.int32)
                eps_raw = max(1, int(round(eps * (raw_max - raw_min) * perceptual_eps_factor)))

                # --- Apply LAB-based smoothing ---
                orig_rgb = img_final[..., rgb_raw_indices].astype(np.float32)
                orig_rgb_01 = (orig_rgb - raw_min) / (raw_max - raw_min)
                lab_orig = color.rgb2lab(orig_rgb_01)
                lab_adv = color.rgb2lab(np.clip(img_scaled, 0.0, 1.0))

                delta_lab = lab_adv - lab_orig
                delta_lab[..., 0] *= 0.1   # L channel minimal
                delta_lab[..., 1:] *= 0.7  # a,b smoother

                # Smooth perturbation spatially
                for ch in range(3):
                    delta_lab[..., ch] = gaussian_filter(delta_lab[..., ch], sigma=smooth_sigma)

                # Reconstruct RGB and map to raw
                rgb_recons = color.lab2rgb(lab_orig + delta_lab)
                rgb_recons_raw = np.clip(np.round(rgb_recons * (raw_max - raw_min) + raw_min),
                                         raw_min, raw_max).astype(raw_dtype)

                # Assign back to raw bands with eps clipping and dithering
                for k, raw_idx in enumerate(rgb_raw_indices):
                    orig_band = img_final[..., raw_idx]
                    adv_band = rgb_recons_raw[..., k]
                    diff = np.clip(adv_band - orig_band, -eps_raw, eps_raw)
                    if dither_scale > 0:
                        noise = (np.random.rand(*diff.shape)-0.5) * 2 * max(1, int(dither_scale*eps_raw))
                        diff = diff + noise.astype(np.int32)
                    img_final[..., raw_idx] = orig_band + diff

                # Clip and convert back
                img_final = np.clip(img_final, raw_min, raw_max).astype(raw_dtype)

                # Save
                fname_tif = f"{base}_{global_ptr+i}_true{int(labels_cpu[i])}_pred{int(preds_cpu[i])}.tif"
                out_path = Path(out_dir) / fname_tif
                tifffile.imwrite(out_path, img_final)
                saved += 1

        global_ptr += images.size(0)

    metrics = {
        "clean_acc": clean_correct / total,
        "adv_acc": adv_correct / total,
        "clean_loss": clean_loss_total / total,
        "adv_loss": adv_loss_total / total,
        "eps": eps,
        "saved": saved,
        "out_dir": os.path.abspath(out_dir)
    }
    return metrics