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

## 4. Training Pipeline

### 4.1 Model Setup

The **ResNet18** model is initialized with pretrained weights. The final fully connected layer is replaced to match the number of classes in the EuroSAT dataset. The model is then moved to the computation device (CPU or GPU) for training.


### 4.2 Loss and Optimizer

The training uses the **cross-entropy loss** function, which is standard in classification projects. 
The **Adam optimizer** is chosen for updating the model weights with a learning rate of 1e-4.


### 4.3 Training Loop

* For each epoch:

  * Forward pass on training set
  * Compute loss and gradients
  * Update weights
  * Evaluate on validation set
* Save the best model based on validation accuracy


### 4.4 Final Evaluation

After training, the best model is loaded and evaluated on the test set to measure final accuracy.

* Test *Accuracy*: 
* Test *F1-score*:
* Test *loss*:
