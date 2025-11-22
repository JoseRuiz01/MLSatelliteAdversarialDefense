import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from src.data.dataloader import get_dataloaders

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if device.type == "cuda" and scaler is not None:
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total



def evaluate(model, dataloader, criterion, device, desc="Validation"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    data_dir="../data/raw",
    batch_size=64,
    epochs=10,
    lr=1e-4,
    model_name="resnet18",
    output_dir="../experiments/checkpoints",
    freeze_backbone=False,
    fine_tune=True,
    device=None,
    early_stopping_patience=5,
):
    """
    Train a classification model on EuroSAT dataset.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # === Data ===
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )
    num_classes = len(class_names)

    # === Model ===
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) 
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model.to(device)

    # === Loss, Optimizer, Scheduler ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # === Setup ===
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.pth")
    best_val_acc = 0.0
    patience_counter = 0

    # === Training Loop ===
    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="Validation")

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best model saved ({val_acc*100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print("⏹️ Early stopping triggered.")
            break

        # Optional fine-tuning halfway
        if fine_tune and freeze_backbone and epoch == epochs // 2:
            print("🔓 Unfreezing backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr / 10)

    # === Final Evaluation ===
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(checkpoint_path))
    _, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"🎯 Test Accuracy: {test_acc*100:.2f}%")

    return model, class_names
