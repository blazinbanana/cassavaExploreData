# Cassava Disease Classification — Data Exploration & Model Development

## Project Goal
The objective of this project is to build a model capable of analyzing images of cassava crops to:

1. Determine whether a plant is healthy.
2. Identify the disease affecting the plant if it is unhealthy.

Correct disease identification is essential because treatment depends on the specific disease detected.

---

## Dataset Exploration Overview
Initial work focused on exploring and preparing the cassava disease image dataset for model training.

The exploration process included:

### 1. Inspecting Image Data
- Checked dataset structure and image properties.
- Verified image formats and dimensions.
- Confirmed class labels and dataset organization.

### 2. Dataset Normalization
Normalization helps neural networks train more effectively by ensuring pixel values are on a consistent scale.

- Normalized image data so that:
  - Mean ≈ 0
  - Standard deviation ≈ 1
- This improves training stability and convergence.

### 3. Class Distribution Analysis
- Counted the number of images per disease class.
- Identified imbalance among classes.
- Some diseases had significantly more images than others.

---

## Handling Class Imbalance
Unbalanced datasets can bias the model toward predicting classes with more samples.

Example issue:
- Fewer images of bacterial blight.
- More images of mosaic and brown streak diseases.
- The model may incorrectly favor predicting more common classes.

### Solution Applied
To address this:

- Classes were balanced using **under-sampling**, reducing samples from larger classes so all classes have similar counts.
- Validation data was also kept nearly balanced.
- Class counts were later verified using a `class_counts` function from the training utilities.

This prevents the model from learning dataset bias rather than disease characteristics.

---

## Preparing Images for PyTorch
Images were preprocessed to make them suitable for PyTorch training.

Steps performed:

1. Converted grayscale images to RGB format.
2. Resized images to a consistent size (e.g., **224 × 224**).
3. Converted images into tensors of pixel values.
4. Applied normalization for improved network performance.
5. Created a transformation pipeline to standardize images before training.
6. Built PyTorch dataloaders for efficient training and validation data loading.

---

## Loss Function: Cross Entropy
For classification tasks, an error metric is needed to measure model performance during training.

### Why Cross Entropy?
Cross entropy measures the difference between:

- Model prediction probabilities
- True class labels

Key points:

- Penalizes incorrect predictions.
- Rewards accurate predictions.
- Lower cross entropy indicates better performance.
- Training minimizes cross entropy using optimization methods.
- Small improvements in predictions reduce loss.
- Works for both binary and multi-class classification problems.

Models with minimized cross entropy typically achieve higher accuracy.

---

## Initial Multiclass Classification Model (CNN)
The initial approach used a Convolutional Neural Network (CNN) to classify crop disease images into **five classes**.

### Objectives Implemented
- Convert grayscale images to RGB
- Resize images
- Normalize data
- Create transformation pipelines
- Build CNN architecture
- Train network for multiclass classification
- Detect overfitting

### CNN Architecture Implemented
1. Convolution Layer 1
2. ReLU activation
3. Max pooling
4. Convolution Layer 2
5. Convolution Layer 3
6. Flatten layer
7. Dropout layer
8. Fully connected layer
9. Output linear layer

### Model Training Setup
- Loss function: Cross Entropy
- Optimizer: Adam
- Device: CUDA (GPU)
- Training executed using a training function from `training.py`.

Training and validation loss and accuracy were plotted, revealing **overfitting**.

---

## Overfitting Observed
The model performed significantly better on training data than validation data, indicating poor generalization to unseen data.

Several mitigation techniques were identified but not implemented at this stage, including:

- Data augmentation
- Dropout tuning
- Regularization
- Early stopping
- Model simplification
- Batch normalization
- Cross-validation
- Increasing dataset size

---

## Transfer Learning Implementation
To improve performance and reduce training time, the project moved to **Transfer Learning**.

### What is Transfer Learning?
Transfer learning uses networks that have already been trained on large image datasets and adapts them to new tasks.

Instead of training an entire model from scratch, only the task-specific layers are trained while the rest of the network acts as a feature extractor.

### Steps Implemented
1. Loaded competition dataset.
2. Downloaded a publicly available pre-trained image classification model.
3. Froze existing network parameters:
   ```python
   params.requires_grad = False
   ```
This prevents pretrained weights from being updated during backpropagation.

4. Replaced the final classification layer

The pretrained model originally classified **1000 classes**, while this task requires only **5 classes**. Therefore:

- The final classification layer was replaced.
- A custom classification head was added.
- Only this new layer is trained.
- The rest of the network performs feature extraction.

This greatly improves training speed and performance.

---

## K-Fold Cross Validation

To further combat overfitting and obtain robust evaluation results, **k-fold cross-validation** was introduced.

### Why K-Fold Cross Validation?

Instead of training and validating on a single split:

- Data is divided into **k subsets**.
- The model is trained multiple times.
- Each subset acts as validation once while others act as training data.

This provides more reliable performance estimates and reduces overfitting risk.

### Implementation Step

A function was built to **reset the custom classification layers** after each fold so every fold begins training from the same starting point.

---

## Current Status

The project has progressed through:

- Dataset exploration and balancing
- Image preprocessing and normalization
- CNN training and overfitting detection
- Transfer learning implementation
- Replacement of final classification layer
- Freezing pretrained model parameters
- K-fold cross-validation setup
- Model reset function for cross-validation cycles

Training now focuses on fine-tuning the transfer learning setup using cross-validation.

---

## Next Steps

Planned improvements include:

- Fine-tuning pretrained layers if necessary
- Applying augmentation and regularization
- Further reducing overfitting
- Hyperparameter tuning
- Final model evaluation
- Preparing deployment pipeline

---

## Summary

The project has evolved from raw dataset exploration to advanced training using transfer learning and cross-validation. The system now leverages pretrained networks for faster training and improved performance, while cross-validation provides more robust evaluation. Further refinement will focus on improving generalization and preparing the model for deployment.
