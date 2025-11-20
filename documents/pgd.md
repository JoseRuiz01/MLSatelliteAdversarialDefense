# Projected Gradient Descent (PGD) Adversarial Attack 

## 1. Introduction

This project implements a **Projected Gradient Descent (PGD)** adversarial attack pipeline designed to evaluate the robustness of image classification models trained on satellite imagery. Adversarial attacks introduce small, carefully crafted perturbations to input images that cause neural networks to misclassify them, while remaining imperceptible or minimally visible to human observers. PGD is one of the most effective iterative attack methods, systematically exploring the worst-case perturbation space around an input to find the most damaging modifications.

The implementation focuses on generating adversarial examples that successfully fool the model while maintaining visual coherence in the resulting images, which is particularly important for multi-band satellite imagery where artifacts can be more noticeable.

---

## 2. PGD Attack Overview

### 2.1 Concept

**Projected Gradient Descent (PGD)** is an iterative adversarial attack that extends the simpler Fast Gradient Sign Method (FGSM) by repeatedly applying small perturbations over multiple iterations. At each step, the attack modifies the input image in the direction that maximally increases the model's loss (for untargeted attacks) or decreases it toward a specific target class (for targeted attacks). Crucially, after each update, the perturbation is projected back into a constrained region defined by an **ε-ball** around the original image, ensuring the modifications remain bounded.

The iterative process can be summarized as:

1. **Compute gradient**: Calculate how the model's loss changes with respect to the input image
2. **Take step**: Move the image in the direction that increases loss
3. **Project**: Ensure the total perturbation doesn't exceed the maximum allowed magnitude (ε)
4. **Repeat**: Iterate multiple times to find stronger adversarial examples

By exploring adversarial directions thoroughly over many iterations, PGD produces more robust attacks compared to single-step methods, making it an excellent benchmark for evaluating model security.

---

## 3. Implementation Approach

This implementation uses a **simplified and direct PGD attack** that prioritizes effectiveness over perceptual sophistication. The goal is to generate strong adversarial examples that successfully cause misclassification, even if this means accepting some visible perturbations.

### 3.1 Core Attack Mechanism

The attack begins by initializing adversarial images with small random noise added to the original inputs. This random start helps the attack explore different perturbation directions and often leads to stronger adversarial examples than starting from the clean image.

At each iteration, the method:

1. **Forward pass**: Runs the current adversarial image through the model
2. **Loss computation**: Calculates the classification loss with respect to the true label
3. **Gradient extraction**: Computes gradients of the loss with respect to the input pixels
4. **Sign-based update**: Uses the sign of the gradient (rather than its magnitude) to determine the direction of perturbation
5. **Epsilon projection**: Clips the total perturbation to remain within the allowed ε-ball
6. **Value clipping**: Ensures pixel values stay in valid ranges (0 to 1 for normalized images)

The sign-based gradient update (similar to FGSM but applied iteratively) provides a good balance between attack strength and computational efficiency.

### 3.2 Attack Parameters

The effectiveness of PGD depends on three key hyperparameters:

* **Epsilon (ε)**: The maximum allowed perturbation magnitude. Larger values produce stronger attacks but more visible changes. Values tested range from 0.03 to 0.1 in normalized image space.

* **Alpha (α)**: The step size for each iteration, typically set as ε/10. This determines how aggressively the attack explores the perturbation space.

* **Iterations**: The number of optimization steps. More iterations generally produce stronger attacks, with 100 iterations providing a good balance between effectiveness and computation time.

### 3.3 Targeted vs Untargeted Attacks

The implementation supports both attack modes:

* **Untargeted attacks** (used in this project): Maximize the loss for the correct class, causing the model to predict any incorrect class
* **Targeted attacks**: Minimize the loss for a specific target class, forcing misclassification toward a chosen category

Untargeted attacks are more realistic for security evaluation, as attackers typically only need to cause incorrect predictions rather than control the specific output.

### 3.4 Adversarial Image Storage

Generated adversarial examples are saved as TIFF files that preserve the attack modifications. The saving process:

1. **Denormalizes** images from the model's input space back to pixel values
2. **Converts** to 8-bit RGB format (0-255 range)
3. **Maintains** the perturbation structure without additional smoothing or processing
4. **Encodes** both true and predicted labels in filenames for easy evaluation

This straightforward approach ensures that the saved adversarial images accurately represent what the model sees, allowing for reliable post-attack evaluation.

---

## 4. Evaluation Pipeline

The complete evaluation workflow processes test datasets in batches and generates comprehensive metrics:

### 4.1 Clean Baseline Measurement

Before applying any attacks, the pipeline:
* Evaluates the model on unmodified test images
* Records accuracy and loss as baseline metrics
* Establishes the model's performance under normal conditions

### 4.2 Adversarial Generation and Testing

For each batch of test images:
* **Initial random perturbation** is added to each image
* **Iterative PGD optimization** runs for the specified number of iterations
* **Gradient-based updates** progressively increase the attack strength
* **Epsilon projection** keeps perturbations within bounds after each step
* **Immediate evaluation** tests the model on the adversarial examples
* **Selective saving** stores a subset of adversarial images for later analysis

### 4.3 Post-Generation Evaluation

After generating adversarial images, a separate evaluation:
* **Reloads** saved adversarial TIFF files from disk
* **Preprocesses** them using the same pipeline as training data
* **Predicts** class labels and confidence scores
* **Updates** filenames to include both true and predicted labels
* **Computes** comprehensive metrics including confusion matrices

This two-stage approach validates that the saved adversarial examples maintain their attack effectiveness.

---

## 5. Design Decisions

The implementation makes several deliberate choices to maximize attack effectiveness:

| Design Choice               | Rationale                                                                                     |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| Random initialization       | Explores diverse perturbation directions; often finds stronger adversarial examples          |
| Sign-based gradients        | Provides consistent step sizes; reduces sensitivity to gradient magnitudes                   |
| Simple projection           | Direct epsilon clipping is efficient and effective; avoids complex constraints               |
| Minimal post-processing     | Preserves attack strength; avoids inadvertently reducing perturbation effectiveness          |
| Direct TIFF saving          | Maintains precise pixel values; ensures saved examples match what the model sees             |
| High epsilon values         | Prioritizes attack success over imperceptibility; tests worst-case model vulnerabilities     |

These choices reflect a focus on **attack effectiveness** for security evaluation rather than generating imperceptible perturbations for deceptive purposes.

---

## 6. Experiments and Results

After generating the final `adv` folder containing the adversarial images, we evaluated the model’s performance using these newly generated samples.

Firstly, we have to understand the main metrics of the attack:

- Each iteration => one small step
- Alpha => step size
- Total iterations => how many steps you take
- Eps => maximum distance you're allowed from starting point

### 6.1. **Experiment on Resnet50 (experiment 2)**


#### 6.1.1. **Experiment 1**
eps: **1/255**      
alpha: **eps/10**
iters: **40**
grad_blur_sigma=**1.0**

* Num images: 100
* Test Accuracy: **53.00%**
* Test Loss: **2.8995**
* Precision: **0.4650**
* Recall: **0.5087**
* F1-score: **0.4610**
*Perturbations are visibles (check e1_img)*


#### 6.1.2. **Experiment 2**
eps: **1/255**      
alpha: **eps/10**
iters: **40**
grad_blur_sigma=**2.0** (*smoother*)
Add low-pass filter to suppress high-frequency artifacts (reduce visible effects of the attack)
Add perceptual loss guidance
(perturbations are more imperceptible) 

* Num images: 100
* Test Accuracy: **30.00%**
* Test Loss: **5.5515**
* Precision: **0.0974**
* Recall: **0.2500**
* F1-score: **0.1402**
*Perturbations are very smooth (check e2_img)*


#### 6.1.3. **Experiment 3**
eps: **4/255**     (*greater perturbation*) 
alpha: **eps/10**
iters: **100**   (*more iterations to optimize*)
grad_blur_sigma=**2.0** 

* Num images: 100
* Test Accuracy: **30.00%**
* Test Loss: **5.4818**
* Precision: **0.0987**
* Recall: **0.2500**
* F1-score: **0.1415**
*No significant changes but with lower performance (check e3_img)*


#### 6.1.4. **Experiment 4**
eps: **4/255**     
alpha: **eps/20**  (*smaller steps*) 
iters: **100**   
grad_blur_sigma=**2.0** 

* Num images: 100
* Test Accuracy: **53.00%**
* Test Loss: **5.5034**
* Precision: **0.0974**
* Recall: **0.2500**
* F1-score: **0.1402**
*No significant changes but with lower performance (check e4_img)*