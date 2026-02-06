# Cassava Disease Classification — Data Exploration 

## Project Goal
The objective of this project is to build a model capable of analyzing images of cassava crops to:

1. Determine whether a plant is healthy.
2. Identify the disease affecting the plant if it is unhealthy.

Correct disease identification is essential because treatment depends on the specific disease detected.

---

## Dataset Exploration Overview
At this stage, work has focused on exploring and preparing the cassava disease image dataset for future model training.

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
- Ensured validation data remains evenly distributed across classes.

This helps prevent the model from learning dataset bias rather than actual disease patterns.

---

## Preparing Images for PyTorch
Images were preprocessed to make them suitable for training in PyTorch.

Steps performed:

1. Converted grayscale images to RGB format.
2. Resized images to a consistent size (e.g., **224 × 224**).
3. Converted images into tensors of pixel values.
4. Applied normalization for improved network performance.

---

## Current Status
So far, the work completed includes:

- Dataset inspection
- Class distribution analysis
- Dataset normalization
- Class balancing via under-sampling
- Image preprocessing for PyTorch compatibility

No model training has been performed yet. This stage strictly covers data exploration and preparation.

---

## Next Steps (Planned)
Future work will involve:

- Model architecture selection
- Model training and validation
- Performance evaluation
- Model deployment or integration

---

## Summary
This stage establishes a clean, balanced, and standardized dataset to ensure reliable model training in subsequent steps.
