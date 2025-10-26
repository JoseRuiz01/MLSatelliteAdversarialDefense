import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.dataloader import get_dataloaders
from torchvision import models
import numpy as np
from src.training.simple_cnn import SimpleCNN



def evaluate_model(model_path, data_dir="data/raw", batch_size=32, model_name="resnet18", device=None):
    """
    Evaluate a trained ResNet model on the test set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader, class_names = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    num_classes = len(class_names)
    
    # === Load model ===
    if model_name == "simplecnn":
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    test_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    return {
        "accuracy": acc,
        "loss": test_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_names": class_names
    }


def plot_confusion_matrix(cm, class_names, figsize=(10,8), normalize=True):
    """
    Plot confusion matrix with Seaborn heatmap
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion Matrix")
    plt.show()
