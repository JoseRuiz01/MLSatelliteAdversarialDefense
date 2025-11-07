import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.training.simple_cnn import SimpleCNN
from src.data.dataloader import get_dataloaders
from src.attacks.utils import DEFAULT_MEAN, DEFAULT_STD
from src.attacks.io import load_tif_image


# -------------------------
# Dataset for raw .tif images (unlabeled)
# -------------------------
class RawFolderDataset(Dataset):
    """
    Simple dataset for .tif images in a folder (no encoded labels).
    """
    def __init__(self, folder, transform=None, pattern="*.tif"):
        from glob import glob
        import os

        self.filepaths = sorted(glob(os.path.join(folder, pattern)))
        if len(self.filepaths) == 0:
            raise FileNotFoundError(f"No images matching {pattern} found in {folder}")

        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img = load_tif_image(path)
        if self.transform:
            img = self.transform(img)
        return img, path


# -------------------------
# Evaluation (same style as training)
# -------------------------
def evaluate(model, dataloader, criterion, device, desc="Evaluation"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# -------------------------
# Evaluate model from checkpoint
# -------------------------
def evaluate_model(
    model_path,
    data_dir="../data/raw",
    batch_size=64,
    model_name="resnet18",
    device=None,
):
    """
    Evaluate a trained model on EuroSAT test set.
    Same metric flow and structure as in training.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # === Load Datasets ===
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )
    num_classes = len(class_names)

    # === Load Model ===
    if model_name.lower() == "simplecnn":
        model = SimpleCNN(num_classes=num_classes)
    elif model_name.lower() == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # === Evaluate ===
    print("\n🧪 Evaluating model on test set...")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, desc="Test")

    # === Compute classification metrics ===
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\nClassification metrics per category:\n")
    print(report)

    # === Plot confusion matrix ===
    plot_confusion_matrix(cm, class_names)

    return {
        "accuracy": test_acc,
        "loss": test_loss,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_names": class_names,
    }


# -------------------------
# Confusion Matrix Plot
# -------------------------
def plot_confusion_matrix(cm, class_names, figsize=(10, 8), normalize=True):
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype("float")
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, where=row_sums != 0)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
