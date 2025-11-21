# Adversarial Training Defense for Deep Learning Models

## 1. Introduction

This project implements an **Adversarial Training** defense mechanism to improve the robustness of image classification models against adversarial attacks. Adversarial training is one of the most effective defense strategies, where a model is explicitly trained on both clean and adversarially perturbed examples during the learning process. By exposing the model to adversarial perturbations during training, it learns to develop more robust decision boundaries that are resistant to small input modifications.

The implementation focuses on hardening a ResNet-50 model trained on satellite imagery against PGD (Projected Gradient Descent) attacks, enabling the model to maintain high accuracy even when confronted with carefully crafted adversarial examples.

---

## 2. Adversarial Training Overview

### 2.1 Concept

**Adversarial Training** is a defensive technique that augments the standard training process by including adversarial examples in each training batch. Rather than training only on clean data, the model learns from a mixture of:

1. **Original clean images**: Standard training samples from the dataset
2. **Adversarial examples**: Perturbed versions generated on-the-fly using an attack algorithm

The key insight is that by continuously exposing the model to worst-case perturbations during training, it learns features that are inherently more stable and less susceptible to adversarial manipulation. This creates a model with decision boundaries that are smoother and more resistant to small input variations.

The training process can be visualized as:

1. **Generate adversarial examples**: For each training batch, create adversarial perturbations using PGD
2. **Combine data**: Merge clean and adversarial examples into a single training batch
3. **Standard training**: Apply conventional supervised learning on the combined dataset
4. **Iterate**: Repeat for multiple epochs until convergence

By treating adversarial examples as additional training data, the model naturally develops robustness without requiring architectural changes or complex defensive mechanisms.

---

## 3. Implementation Approach

This implementation integrates PGD attack generation directly into the training loop, creating a seamless adversarial training pipeline that balances model accuracy on clean data with robustness against attacks.

### 3.1 Core Defense Mechanism

The adversarial training process extends standard supervised learning with an adversarial example generation step:

**During each training iteration:**

1. **Load clean batch**: Sample a batch of clean images and labels from the training dataset
2. **Generate adversarial examples**: Apply PGD attack to create perturbed versions of the clean images
3. **Data augmentation**: Concatenate clean and adversarial images to form an augmented batch (2× original size)
4. **Forward pass**: Process the combined batch through the model
5. **Loss computation**: Calculate cross-entropy loss on both clean and adversarial examples
6. **Backpropagation**: Update model weights based on the combined loss
7. **Optimization**: Apply gradient descent to minimize loss on both data types

This approach ensures the model simultaneously learns to:
- Correctly classify clean, unperturbed images
- Resist adversarial perturbations by maintaining correct predictions under attack

### 3.2 PGD Attack Integration

The adversarial examples are generated using a sophisticated PGD attack with perceptual enhancements:

**Attack Configuration:**
```python
PGDConfig(
    eps=2/255,              # Perturbation budget
    alpha=eps/10,           # Step size per iteration
    iters=40,              # Number of attack iterations
    grad_blur_sigma=2.0,    # Gradient smoothing
    device=device
)
```

**Perceptual Enhancements:**
- **Gradient smoothing**: Gaussian blur reduces high-frequency artifacts
- **Low-pass filtering**: Suppresses sharp color transitions in perturbations
- **Saliency masking**: Focuses perturbations on less perceptually important regions
- **Perceptual loss penalty**: Constrains perturbations to be less noticeable

These enhancements create adversarial examples that are both challenging for the model and visually coherent, ensuring the defense generalizes to realistic attack scenarios.

### 3.3 Training Parameters

The effectiveness of adversarial training depends on several key hyperparameters:

* **Perturbation budget (ε)**: Set to 2/255, allowing moderate perturbations that test model robustness without being trivially visible

* **Attack iterations**: 40 iterations per PGD attack ensure strong adversarial examples that thoroughly explore the perturbation space

* **Learning rate**: 5e-5, lower than standard training to account for the increased difficulty of learning from adversarial data

* **Training epochs**: 10 epochs provide sufficient exposure to adversarial examples while avoiding overfitting

* **Batch size**: 64 images per batch, doubled to 64 when combining clean and adversarial examples

* **Gradient blur sigma**: 2.0, smoothing gradients during attack generation to reduce visible artifacts

### 3.4 Model Architecture

The defense is applied to a **ResNet-50** architecture:

* **Backbone**: Pre-trained ResNet-50 from ImageNet for transfer learning benefits
* **Final layer**: Customized fully connected layer for the specific number of classes (4 for EuroSAT subset)
* **Optimizer**: Adam optimizer with low learning rate for stable adversarial training
* **Loss function**: Standard cross-entropy loss applied to combined clean and adversarial batches

The pre-trained backbone provides strong feature representations, which adversarial training then refines to be more robust against perturbations.

---

## 4. Training Pipeline

The complete adversarial training workflow follows a structured approach:

### 4.1 Model Initialization

The pipeline begins by:
* **Loading pre-trained weights** from a clean model trained on standard data
* **Configuring attack parameters** for PGD adversarial example generation
* **Setting up optimizer** with appropriate learning rate for adversarial training
* **Initializing loss function** (cross-entropy) for both clean and adversarial samples

Starting from a pre-trained clean model provides a strong foundation, allowing adversarial training to refine existing features rather than learning from scratch.

### 4.2 Adversarial Training Loop

For each training epoch:

**Batch Processing:**
1. **Sample clean data**: Load a batch of original training images
2. **Generate adversarial examples**: Apply PGD attack to create perturbed versions
   - Model is temporarily set to evaluation mode during attack generation
   - Gradients are computed with respect to inputs (not weights)
   - Perturbations are projected to stay within ε-ball
3. **Combine datasets**: Concatenate clean and adversarial images into a single batch
4. **Model training**: Switch model back to training mode
5. **Forward pass**: Process combined batch through the network
6. **Compute loss**: Calculate cross-entropy loss on all samples
7. **Backpropagation**: Update model weights to minimize combined loss
8. **Track metrics**: Record loss and accuracy for monitoring

**Epoch Monitoring:**
- Display training loss and accuracy after each epoch
- Track convergence to ensure the model is learning effectively
- Identify potential overfitting or underfitting issues

### 4.3 Model Checkpointing

After training completion:
* **Save model weights** to disk with a descriptive name (e.g., `resnet50_adv_trained.pth`)
* **Preserve architecture state** including final layer modifications
* **Enable reproducibility** by saving all learned parameters

The saved model can then be deployed for inference or further evaluated against various attack scenarios.

---

## 5. Evaluation Strategy

To comprehensively assess the adversarial training defense, we employ a two-phase evaluation:

### 5.1 Clean Data Evaluation

**Purpose**: Verify the model maintains good performance on unperturbed images

**Process:**
1. Load the adversarially trained model
2. Evaluate on standard test set without any perturbations
3. Measure accuracy, precision, recall, and F1-score
4. Compare against pre-training baseline to quantify accuracy trade-off

**Key Metric**: Clean test accuracy should ideally remain close to the original model's performance, with minimal degradation (< 5% drop acceptable).

### 5.2 Adversarial Robustness Evaluation

**Purpose**: Measure defense effectiveness against adversarial attacks

**Process:**
1. Load pre-generated adversarial test images from the `adv` folder
2. These images were created using strong PGD attacks on a clean model
3. Pass adversarial examples through the adversarially trained model
4. Measure accuracy, precision, recall, and F1-score on attacked images
5. Generate confusion matrix to identify class-specific vulnerabilities

**Key Metric**: Adversarial test accuracy should show significant improvement over the clean model's performance on the same adversarial examples.

### 5.3 Success Criteria

A successful adversarial training defense demonstrates:

* **Robustness gain**: Substantial increase in adversarial accuracy (target: > 20% improvement)
* **Clean accuracy preservation**: Minimal degradation on clean data (target: < 5% loss)
* **Generalization**: Improved robustness across all classes, not just specific categories
* **Attack resistance**: Maintained performance against the attack configuration used during training

---

## 6. Design Decisions

The implementation makes several strategic choices to optimize defense effectiveness:

| Design Choice                    | Rationale                                                                                              |
|----------------------------------|--------------------------------------------------------------------------------------------------------|
| On-the-fly attack generation     | Ensures diverse adversarial examples; prevents model from memorizing specific perturbations           |
| Combined batch training          | Balances clean and adversarial performance; prevents catastrophic forgetting of clean data            |
| Perceptual attack enhancements   | Creates realistic adversarial examples; ensures defense works against practical attack scenarios      |
| Pre-trained initialization       | Leverages transfer learning; reduces training time and improves feature quality                       |
| Low learning rate (5e-5)         | Provides stable updates; prevents model from overfitting to adversarial examples                      |
| Gradient smoothing during attack | Reduces high-frequency perturbations; makes defense more practical for real-world scenarios           |
| Clean + adversarial mix          | Prevents model from becoming too conservative; maintains discriminative power on normal data          |

These choices reflect a focus on **practical robustness** against realistic attack scenarios while maintaining competitive accuracy on clean data.

---

## 7. Experiments and Results

After completing adversarial training on the ResNet-50 model with the EuroSAT satellite imagery dataset, we evaluated the defense mechanism's effectiveness against PGD attacks.

**Training Configuration:**
- Base model: ResNet-50 pre-trained on ImageNet
- Training epochs: 10
- Batch size: 64 (96 combined with adversarial examples)
- Learning rate: 5e-5
- Optimizer: Adam

**Attack Configuration (Training & Evaluation):**
- Perturbation budget (ε): 2/255
- Step size (α): ε/10
- Attack iterations: 40
- Gradient blur sigma: 2.0
- Additional: Low-pass filtering, saliency masking, perceptual loss

### 7.1 Baseline Performance (Clean Model)

**On Clean Test Data:**
- Accuracy: 99.88%
- Loss: 0.0060
- Precision: 0.9988
- Recall: 0.9989
- F1-score: 0.9988

**On Adversarial Test Data (experiment 2 for performance):**
- Accuracy: 30.00%
- Loss: 5.5515
- Precision: 0.0974
- Recall: 0.2500
- F1-score: 0.1402

*The clean model shows severe vulnerability to adversarial attacks, with accuracy dropping from 99% to 30%.*

---

### 7.2 Adversarial Training Results

#### Experiment 1: Standard Adversarial Training

**Configuration:**
- epochs: 5

**Results - Clean Test Data:**
- Accuracy: **99.71%**
- Loss: **0.0124**
- Precision: **0.9971**
- Recall: **0.9970**
- F1-score: **0.9971**

**Results - Adversarial Test Data:**
- Accuracy: **32.00%**
- Loss: **5.0033**
- Precision: **0.3514**
- Recall: **0.2750**
- F1-score: **0.1897**

---

#### Experiment 2: Adversarial Training with Data Augmentation

**Configuration:**
- epochs: 5
- Data augmentation:
    - ResizedCrop
    - HorizontalFlip
    - VerticalFlip
    - Rotation
    - Brightness
    - Contrast
    - Saturation
    - Hue


**Results - Clean Test Data:**
- Accuracy: **[TO BE FILLED AFTER TRAINING]**%
- Loss: **[TO BE FILLED]**
- Precision: **[TO BE FILLED]**
- Recall: **[TO BE FILLED]**
- F1-score: **[TO BE FILLED]**

**Results - Adversarial Test Data:**
- Accuracy: **[TO BE FILLED]**%
- Loss: **[TO BE FILLED]**
- Precision: **[TO BE FILLED]**
- Recall: **[TO BE FILLED]**
- F1-score: **[TO BE FILLED]**


---
