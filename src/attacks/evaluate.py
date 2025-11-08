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
from .pgd import pgd_attack_batch, PGDConfig
from .utils import extract_mean_std, unnormalize, DEFAULT_MEAN, DEFAULT_STD
from .io import save_adversarial_tif


def evaluate_pgd(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    eps: float = 0.005,
    alpha: Optional[float] = None,
    iters: int = 50,
    out_dir: str = "../data/adversarial/pgd",
    save_every: int = 20,
    max_save: Optional[int] = 200,
    targeted: bool = False,
    target_class: Optional[int] = None,
    perceptual_eps_factor: float = 0.45,
    smooth_sigma: float = 1.0,
    grad_mask_fraction: float = 0.25,
    dither_scale: float = 0.3,
) -> Dict:
    """
    Top-level evaluation loop that:
      - computes clean accuracy & loss
      - runs PGD attack per-batch
      - computes adversarial accuracy & loss
      - periodically saves adversarial TIFFs using perceptual mapping

    Returns: metrics dict
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    mean_t, std_t = extract_mean_std(dataloader)
    if mean_t is None or std_t is None:
        import torch as _t
        mean_t = _t.tensor(DEFAULT_MEAN).view(-1, 1, 1)
        std_t = _t.tensor(DEFAULT_STD).view(-1, 1, 1)

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

    pgd_conf = PGDConfig(
        eps=eps,
        alpha=alpha,
        iters=iters,
        small_step_fraction=0.2,
        grad_mask_fraction=grad_mask_fraction,
        grad_blur_sigma=1.0,
        smooth_perturb_sigma=smooth_sigma,
        random_dither=True,
        dither_scale=dither_scale,
        device=device,
    )

    for images, labels in tqdm(dataloader, desc=f"PGD eps={eps}", leave=False):
        batch_idx += 1
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        total += batch_size

        with torch.no_grad():
            out = model(images)
            loss = criterion(out, labels)
            _, preds = out.max(1)
            clean_correct += (preds == labels).sum().item()
            clean_loss_total += loss.item() * batch_size

        target_labels = None
        if targeted:
            if target_class is None:
                with torch.no_grad():
                    probs = torch.softmax(out, dim=1)
                    target_labels = probs.argsort(dim=1)[:, -2]
            else:
                target_labels = torch.full_like(labels, fill_value=int(target_class), device=device)

        adv_images = pgd_attack_batch(
            model,
            images,
            labels,
            config=pgd_conf,
            targeted=targeted,
            target_labels=target_labels,
        )

        with torch.no_grad():
            out_adv = model(adv_images)
            loss_adv = criterion(out_adv, labels)
            _, preds_adv = out_adv.max(1)
            adv_correct += (preds_adv == labels).sum().item()
            adv_loss_total += loss_adv.item() * batch_size

        if (batch_idx % save_every == 0) and (max_save is None or saved < max_save):
            adv_cpu = adv_images.cpu()
            labels_cpu = labels.cpu().numpy()

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
                    # Save first
                    out_path = save_adversarial_tif(
                        adv_img_tensor=adv_cpu,
                        index_in_batch=i,
                        orig_path=orig_name,
                        mean_t=mean_t,
                        std_t=std_t,
                        labels_cpu=labels_cpu,
                        out_dir=out_dir,
                        eps=eps,
                        perceptual_eps_factor=perceptual_eps_factor,
                        smooth_sigma=smooth_sigma,
                        dither_scale=dither_scale,
                    )

                    img = tifffile.imread(out_path)
                    if img.ndim == 2:
                        img = np.stack([img]*3, axis=-1)
                    elif img.ndim == 3 and img.shape[2] >= 4:
                        img = img[..., [3,2,1]]  # select RGB

                    img = Image.fromarray(img.astype(np.uint8))
                    preprocess = transforms.Compose([
                        transforms.Resize((64,64)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean_t.squeeze().tolist(), std=std_t.squeeze().tolist())
                    ])
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out_pred = model(img_tensor)
                        pred_class = out_pred.argmax(1).item()

                    base = Path(orig_name).stem
                    true_label = int(labels_cpu[i])
                    new_fname = f"{base}_true{true_label}_pred{pred_class}.tif"
                    new_path = Path(out_dir) / new_fname
                    os.rename(out_path, new_path)
                    saved += 1

                except Exception as e:
                    print(f"Warning: failed to save or rename adversarial image for {orig_name}: {e}")


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
