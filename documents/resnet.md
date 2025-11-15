# ResNet Basic Model and Training for Image Classification

## 1. Introduction

This document describes the setup and training of a baseline **ResNet** model for classifying satellite images from the EuroSAT dataset. The trained model will later be used to study adversarial attacks on image classifiers.

EuroSAT images consist of RGB bands in `.tif` format organized in class folders, representing different land  categories.

---

## 2. Model Overview: ResNet18

**ResNet (Residual Network)** is a convolutional neural network designed to train very deep models by using residual connections that mitigate the vanishing gradient problem.

We use **ResNet18**, which consists of:

* An initial convolutional layer followed by max pooling.
* Four residual blocks with increasing feature channels.
* A fully connected layer adapted to the number of classes in EuroSAT (10 classes).

Key advantages:

* Deep network with skip connections ensures stable training.
* Pretrained weights (from *ImageNet*) can be fine-tuned to improve performance.

---

## 3. Dataset Handling

### 3.1 Dataset Structure

The dataset is stored in `data/raw` with subfolders for each class:

```
data/raw/
├── Forest/
├── Highway/
├── Industrial/
└── ...
```

### 3.2 Custom Dataset with Rasterio

We use `rasterio` to read `.tif` images. Only the first 3 bands (RGB) are loaded and converted to `uint8`:

### 3.3 DataLoader

We split the dataset into train, validation, and test sets (default 70/15/15). DataLoader applies the following transforms:

* Resize images to 64x64
* **NO Data Augmentation** to later study attacks modifying the data
* Conversion to tensor and normalization using per-channel mean and std

---

Here’s an updated and clear version of your **Training Pipeline** section that now includes both **ResNet18** and **SimpleCNN**:

---

## 4. Training Pipeline

### 4.1 Model Setup

Two model architectures are supported in this project:

* **ResNet18**: A deep convolutional neural network pretrained on ImageNet. The final fully connected layer is replaced to match the number of classes in the EuroSAT dataset. This model generally achieves high accuracy but requires more computation.

* **SimpleCNN**: A lightweight custom convolutional network designed for faster training and experimentation. It consists of a few convolutional and fully connected layers, making it easier to train on limited hardware or smaller datasets while still capturing essential spatial patterns.

Both models are automatically moved to the available computation device (CPU or GPU) for training.

---

### 4.2 Loss and Optimizer

The training process uses the **Cross-Entropy Loss**, which measures the difference between predicted and true class probabilities — a standard choice for multi-class classification tasks.
The **Adam optimizer** is used for parameter updates, providing adaptive learning rates for stable convergence. The learning rate is set to **1e-4** by default but can be adjusted based on performance.

---

### 4.3 Training Loop

For each training epoch:

1. Perform a forward pass through the training data.
2. Compute the loss and perform backpropagation to calculate gradients.
3. Update the model weights using the optimizer.
4. Evaluate the model on the validation dataset to track performance.
5. Save the model checkpoint whenever the validation accuracy improves — ensuring that the best version of the model is preserved for testing and deployment.


### 4.4 Final Evaluation

After training, the best model is loaded and evaluated on the test set to measure final accuracy.

#### 4.4.1. **Experiment 1**
**ResNet18** (Forest, Highway, SeaLake)
* Device: **macOS M1 CPU**
* Test Accuracy: **99.14%**
* Test Loss: **0.0220**
* Precision: **0.9912**
* Recall: **0.9917**
* F1-score: **0.9914**

#### 4.4.2. **Experiment 2**
**ResNet50** (AnnualCrop, Forest, Residential, River)
Device: **Kaggle accelerator GPU P100**
* Test Accuracy: **99.88%**
* Test Loss: **0.0060**
* Precision: **0.9988**
* Recall: **0.9989**
* F1-score: **0.9988**

#### 4.4.3. **Experiment 3**
**ResNet50** (AnnualCrop, Forest, Highway, Industrial, Residential, River)
Device: **Kaggle accelerator GPU P100**
* Test Accuracy: **9**
* Test Loss: **0.02**
* Precision: **0.9**
* Recall: **0.9**
* F1-score: **0.9**