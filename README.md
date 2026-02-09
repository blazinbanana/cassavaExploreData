# Cassava Disease Classification — Data Exploration & Initial Model Training

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
Cross entropy is widely used for classification problems because it measures the difference between:

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

## Multiclass Classification Task
The goal is to classify crop disease images into **five classes** using a Convolutional Neural Network (CNN).

Key objectives implemented:

- Convert grayscale images to RGB
- Resize images
- Normalize data
- Create transformation pipelines
- Build a CNN architecture
- Train the network for multiclass classification
- Detect overfitting

---

## CNN Architecture Implemented
A Convolutional Neural Network was built with the following structure:

1. **Convolution Layer 1**
2. **ReLU Activation**
   - Introduces non-linearity.
   - Helps remove noise while retaining strong features.
3. **Max Pooling**
   - Reduces spatial dimensions.
   - Decreases computation.
   - Helps prevent overfitting.
   - Retains prominent features.

4. **Convolution Layer 2**
5. **Convolution Layer 3**
6. **Flatten Layer**
7. **Dropout Layer**
8. **Fully Connected (Linear) Layer**
9. **Output Linear Layer**

---

## Model Training
Training setup:

- Loss function: **Cross Entropy**
- Optimizer: **Adam**
- Computation device: **CUDA (GPU)**
- Training executed using a training function from `training.py`.

After training:
- Training and validation loss curves were plotted.
- Training and validation accuracy were also plotted.

These plots revealed the presence of **overfitting**, where training performance improves while validation performance degrades.

---

## Overfitting Observed
The model performed significantly better on training data compared to validation data, indicating that it memorized training samples rather than generalizing to unseen data.

---

## Potential Solutions to Overfitting
Several strategies were identified but **not yet implemented**:

- **Data Augmentation**  
  Generate new training samples via rotations, flips, and scaling.

- **Dropout Layers**  
  Randomly deactivate neurons during training to reduce reliance on specific features.

- **Regularization (L1/L2)**  
  Penalize overly complex models.

- **Early Stopping**  
  Stop training once validation loss begins to increase.

- **Reduce Model Complexity**  
  Use fewer layers or smaller layers.

- **Use More Data**  
  Increase dataset size if possible.

- **Batch Normalization**  
  Improve training stability and sometimes reduce overfitting.

- **Cross-validation**  
  Validate performance across different data splits.

None of these mitigation strategies have yet been applied.

---

## Current Status
The project has progressed from data exploration to initial CNN training.

Completed work includes:

- Dataset inspection and balancing
- Image preprocessing and normalization
- Data transformation pipelines
- Dataloader creation
- CNN architecture implementation
- Model training using cross entropy loss
- Detection of overfitting

No overfitting mitigation strategies have been applied yet.

---

## Next Steps
Planned next steps include:

- Apply overfitting mitigation techniques
- Improve model generalization
- Tune model architecture and training parameters
- Re-train and evaluate performance
- Prepare the model for deployment

---

## Summary
The project has successfully moved beyond dataset preparation into initial CNN training. However, the current model suffers from overfitting, and the next development phase will focus on improving generalization before deployment.
