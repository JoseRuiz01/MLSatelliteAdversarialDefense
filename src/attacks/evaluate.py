# src/attacks/evaluate.py
import os
import re
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.training.simple_cnn import SimpleCNN
import tifffile

try:
    from src.data.dataloader import EuroSATDataset, compute_mean_std, get_dataloaders
except Exception:
    EuroSATDataset = None
    compute_mean_std = None
    get_dataloaders = None


class AdvFolderDataset(Dataset):
    """
    Dataset for adversarial images saved as TIF with true label encoded in filename.
    """

    def __init__(self, folder, transform=None, pattern="*.tif", class_names=None, data_dir=None):
        self.folder = folder
        self.pattern = pattern
        self.transform = transform
        self.filepaths = sorted(glob.glob(os.path.join(folder, pattern)))
        if len(self.filepaths) == 0:
            raise FileNotFoundError(f"No images matching {pattern} found in {folder}")

        self.class_names = class_names
        if self.class_names is None and data_dir is not None and get_dataloaders is not None:
            try:
                _, _, _, self.class_names = get_dataloaders(data_dir=data_dir, batch_size=1)
            except Exception:
                self.class_names = None

        self.labels_parsed = None

    def __len__(self):
        return len(self.filepaths)

    def _parse_label_from_filename(self, fname):
        base = os.path.basename(fname)
        m = re.search(r"true[_\-]?(\d+)", base, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"true[_\-\.\s]?([A-Za-z0-9]+)", base, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        raise ValueError(f"Could not parse true label from filename '{base}'.")

    def parse_labels(self):
        labels = []
        for fp in self.filepaths:
            raw = self._parse_label_from_filename(fp)
            if isinstance(raw, int):
                labels.append(raw)
            else:
                if self.class_names is not None:
                    if raw in self.class_names:
                        labels.append(self.class_names.index(raw))
                    else:
                        lowered = [c.lower() for c in self.class_names]
                        labels.append(lowered.index(raw.lower()))
                else:
                    raise ValueError(f"Parsed class name '{raw}' but class_names not provided.")
        self.labels_parsed = labels
        return labels

    def __getitem__(self, idx):
        if self.labels_parsed is None:
            self.parse_labels()

        p = self.filepaths[idx]
        img = tifffile.imread(p) 

        if img.ndim == 2: 
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3:
            if img.shape[0] >= 4:
                img = np.transpose(img, (1,2,0))
            # Selecciona solo RGB
            if img.shape[2] >= 4:
                img = img[..., [3,2,1]]

        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)
        label = self.labels_parsed[idx]
        return img, label



def get_mean_std(data_dir, sample_size=2000, device="cpu"):
    """
    Use the project's dataloader helper to compute mean/std if available.
    """
    if EuroSATDataset is None or compute_mean_std is None:
        
        return [0.3443, 0.3803, 0.4082], [0.1573, 0.1309, 0.1198]

    base_ds = EuroSATDataset(data_dir, transform=transforms.ToTensor())
    mean_t, std_t = compute_mean_std(base_ds, sample_size=sample_size)

    if torch.is_tensor(mean_t):
        mean = mean_t.tolist()
    else:
        mean = list(mean_t)
    if torch.is_tensor(std_t):
        std = std_t.tolist()
    else:
        std = list(std_t)
    return mean, std


def evaluate_adv(
    adv_folder,
    model_path,
    data_dir=None,
    batch_size=32,
    model_name="resnet18",
    device=None,
    mean_std_sample_size=2000,
    image_pattern="*.png"
):
    """
    Evaluate a trained model on adversarial images saved in adv_folder.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_dir is None:
        mean, std = [0.3443, 0.3803, 0.4082], [0.1573, 0.1309, 0.1198]
    else:
        mean, std = get_mean_std(data_dir, sample_size=mean_std_sample_size, device=device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    class_names = None
    if data_dir is not None and get_dataloaders is not None:
        try:
            _, _, _, class_names = get_dataloaders(data_dir=data_dir, batch_size=1)
        except Exception:
            class_names = None

    adv_ds = AdvFolderDataset(adv_folder, transform=transform, pattern=image_pattern, class_names=class_names, data_dir=data_dir)
    adv_loader = DataLoader(adv_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    if class_names is None:
        class_names = [str(i) for i in range(max(adv_ds.parse_labels()) + 1)]
    num_classes = len(class_names)
    
    if model_name == "simplecnn":
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in adv_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    if total_samples == 0:
        raise RuntimeError("No samples were evaluated (total_samples == 0)")

    test_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)

    return {
        "accuracy": acc,
        "loss": test_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_names": class_names,
        "num_images": len(adv_ds)
    }


def plot_confusion_matrix(cm, class_names, figsize=(10,8), normalize=True):
    """
    Plot confusion matrix with Seaborn heatmap
    """
    if normalize:
        with np.errstate(all='ignore'):
            cm_norm = cm.astype('float')
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm_norm, row_sums, where=row_sums!=0)
    else:
        cm_norm = cm

    plt.figure(figsize=figsize)
    sns.heatmap(cm_norm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
