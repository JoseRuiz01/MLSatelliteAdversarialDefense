from typing import Optional, Dict
import os
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import tifffile
from PIL import Image
from torchvision import transforms
from .pgd import pgd_attack_batch, smooth_perturbation, PGDConfig
from .utils import extract_mean_std, unnormalize, DEFAULT_MEAN, DEFAULT_STD
from .io import save_adversarial_tif


def evaluate_pgd(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    eps: float = 2.0,       
    alpha: Optional[float] = None,
    iters: int = 20,
    out_dir: str = "../data/adv/adv_1",
    save_every: int = 1,
    max_save: Optional[int] = 200,
    targeted: bool = False,
    target_class: Optional[int] = None,
    smooth_sigma: float = 0.8, 
) -> Dict:
    """
    Evaluation function with proper space handling.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    # Extract normalization parameters
    mean_t, std_t = extract_mean_std(dataloader)
    if mean_t is None or std_t is None:
        mean_t = torch.tensor(DEFAULT_MEAN).view(-1, 1, 1)
        std_t = torch.tensor(DEFAULT_STD).view(-1, 1, 1)

    criterion = nn.CrossEntropyLoss()
    total = 0
    clean_correct = 0
    adv_correct = 0
    clean_loss_total = 0.0
    adv_loss_total = 0.0
    saved = 0
    batch_idx = 0
    global_ptr = 0
    
    # Lists to store all predictions for metrics
    all_clean_preds = []
    all_adv_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc=f"PGD eps={eps}", leave=False):
        batch_idx += 1
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        total += batch_size

        # Clean evaluation
        with torch.no_grad():
            out = model(images)
            loss = criterion(out, labels)
            _, preds = out.max(1)
            clean_correct += (preds == labels).sum().item()
            clean_loss_total += loss.item() * batch_size
            all_clean_preds.extend(preds.cpu().numpy())

        # Target selection
        target_labels = None
        if targeted:
            if target_class is None:
                with torch.no_grad():
                    probs = torch.softmax(out, dim=1)
                    target_labels = probs.argsort(dim=1)[:, -2]
            else:
                target_labels = torch.full_like(labels, fill_value=int(target_class))

        # Generate adversarial examples using FIXED attack
        adv_images = pgd_attack_batch(
            model=model,
            images=images,
            labels=labels,
            mean=mean_t,
            std=std_t,
            eps=eps,  # In 0-255 scale
            alpha=alpha,
            iters=iters,
            targeted=targeted,
            target_labels=target_labels,
        )
        
        # Optional: Smooth perturbations to reduce visibility
        if smooth_sigma > 0:
            with torch.no_grad():
                # Get perturbation in pixel space
                orig_pixel = images * std_t.to(device) + mean_t.to(device)
                adv_pixel = adv_images * std_t.to(device) + mean_t.to(device)
                pert_pixel = adv_pixel - orig_pixel
                
                # Smooth
                pert_np = pert_pixel.cpu().numpy()
                pert_smooth = smooth_perturbation(pert_np, sigma=smooth_sigma)
                pert_smooth = torch.from_numpy(pert_smooth).to(device)
                
                # Apply smoothed perturbation
                adv_pixel_smooth = torch.clamp(orig_pixel + pert_smooth, 0, 1)
                adv_images = (adv_pixel_smooth - mean_t.to(device)) / std_t.to(device)

        # Adversarial evaluation
        with torch.no_grad():
            out_adv = model(adv_images)
            loss_adv = criterion(out_adv, labels)
            _, preds_adv = out_adv.max(1)
            adv_correct += (preds_adv == labels).sum().item()
            adv_loss_total += loss_adv.item() * batch_size
            all_adv_preds.extend(preds_adv.cpu().numpy())
            
        all_labels.extend(labels.cpu().numpy())

        # Save images
        if (batch_idx % save_every == 0) and (max_save is None or saved < max_save):
            adv_cpu = adv_images.cpu()
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds_adv.cpu().numpy()

            for i in range(batch_size):
                if max_save is not None and saved >= max_save:
                    break

                try:
                    if hasattr(dataloader.dataset, "indices"):
                        idx = dataloader.dataset.indices[global_ptr + i]
                        orig_dataset = dataloader.dataset.dataset
                    else:
                        idx = global_ptr + i
                        orig_dataset = dataloader.dataset

                    orig_name = orig_dataset.samples[idx][0]
                except Exception:
                    orig_name = f"sample_{global_ptr + i}.tif"

                try:
                    out_path = save_adversarial_tif(
                        adv_img_tensor=adv_cpu,
                        index_in_batch=i,
                        orig_path=orig_name,
                        mean_t=mean_t,
                        std_t=std_t,
                        labels_cpu=labels_cpu,
                        out_dir=out_dir,
                    )

                    # Rename with prediction
                    base = Path(orig_name).stem
                    true_label = int(labels_cpu[i])
                    pred_class = int(preds_cpu[i])
                    new_fname = f"{base}_true{true_label}_pred{pred_class}.tif"
                    new_path = Path(out_dir) / new_fname
                    os.rename(out_path, new_path)
                    saved += 1

                except Exception as e:
                    print(f"Warning: failed to save image: {e}")

        global_ptr += batch_size

    metrics = {
        "clean_acc": clean_correct / total if total > 0 else 0.0,
        "adv_acc": adv_correct / total if total > 0 else 0.0,
        "clean_loss": clean_loss_total / total if total > 0 else 0.0,
        "adv_loss": adv_loss_total / total if total > 0 else 0.0,
        "eps": eps,
        "saved": saved,
        "out_dir": os.path.abspath(out_dir),
    }
    return metrics
