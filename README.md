# Cassava Disease Classification
> A deep learning pipeline for multi-class plant disease detection using transfer learning, k-fold cross-validation, and callback-enhanced training.

---

## Project Objective

Develop a robust deep learning model to classify cassava plant images into:

| Label | Description |
|-------|-------------|
| `0` | Cassava Bacterial Blight |
| `1` | Cassava Brown Streak Disease |
| `2` | Cassava Green Mottle |
| `3` | Cassava Mosaic Disease |
| `4` | Healthy |

Accurate classification enables targeted agricultural intervention at scale.

---

## Pipeline Overview

```
Raw Images
    │
    ▼
Dataset Exploration & Validation
    │
    ▼
Preprocessing Pipeline (Resize → Normalize → Tensor)
    │
    ▼
Class Imbalance Handling (Under-sampling)
    │
    ▼
Baseline CNN → Overfitting Analysis
    │
    ▼
Transfer Learning (Pretrained Backbone)
    │
    ▼
K-Fold Cross Validation
    │
    ▼
Callback-Enhanced Training Loop
    │
    ▼
Inference → Submission CSV
```

---

## 1. Dataset Exploration & Validation

- Verified dataset structure and class organization
- Inspected image formats, dimensions, and consistency
- Validated label mappings across all classes
- Computed per-class image counts and identified **significant class imbalance**

---

## 2. Data Preprocessing Pipeline

### Transformations Applied

```python
transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # Grayscale → RGB
    transforms.Resize((224, 224)),                  # Standardize dimensions
    transforms.ToTensor(),                          # Convert to PyTorch tensor
    transforms.Normalize(mean=[...], std=[...])     # Mean ≈ 0, Std ≈ 1
])
```

> Pipelines are integrated into PyTorch `Dataset` and `DataLoader` for seamless batching.

---

## 3. Handling Class Imbalance

**Problem:** Skewed class distribution leads to biased model predictions.

**Solution:**
- Applied **under-sampling** on majority classes
- Maintained a near-balanced validation set
- Verified distributions post-sampling using utility functions

---

## 4. Baseline Model — Custom CNN

### Architecture

```
Input (224×224×3)
    │
Conv2D → ReLU → MaxPool
    │
Conv2D → Conv2D
    │
Flatten → Dropout
    │
Fully Connected → Output (5 classes)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | Cross Entropy |
| Optimizer | Adam |
| Device | CUDA GPU |

### Outcome

- High training accuracy
- Lower validation performance — **overfitting detected**

---

## 5. Overfitting Analysis

**Observation:** Divergence between training and validation loss curves.

**Mitigation Strategies Identified:**

- Data augmentation
- Regularization (Dropout tuning, weight decay)
- Early stopping
- Model simplification
- Batch normalization
- Cross-validation

---

## 6. Transfer Learning

Leveraged a pretrained CNN (ImageNet-trained) as a frozen feature extractor.

### Implementation

```python
# Freeze backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Replace classification head
model.fc = nn.Linear(in_features, num_classes=5)
```

### Benefits

| Benefit | Detail |
|---------|--------|
| Faster convergence | Pretrained weights provide strong initialization |
| Better feature extraction | Rich ImageNet representations |
| Reduced training cost | Only classification head is trained |

---

## 7. K-Fold Cross Validation

**Motivation:** Improve evaluation robustness and reduce variance from a single train/val split.

### Implementation

```
Dataset
 ├── Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]
 ├── Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]
 ├── Fold 3: [TRN] [TRN] [VAL] [TRN] [TRN]
 ├── Fold 4: [TRN] [TRN] [TRN] [VAL] [TRN]
 └── Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]
```

> **Note:** The classification head is reset after each fold to ensure training consistency.

---

## 8. Training Callbacks

### 8.1 Early Stopping

```
Monitors  → Validation Loss
Trigger   → No improvement for N epochs (patience)
Effect    → Prevents overfitting, reduces training time
```

### 8.2 Model Checkpointing

```
Trigger   → Validation loss improves
Saves     → Best model weights per epoch
Effect    → Retains optimal state, avoids late-epoch degradation
```

### 8.3 Learning Rate Scheduling

```
Strategy  → Dynamic decay during training
Controls  → Step size (epoch interval) + decay factor
Optimizer → Adam
Effect    → Large updates early, fine-tuned updates near convergence
```

---

## 9. Final Training Pipeline

The optimized end-to-end pipeline:

```
Balanced Dataset
    + Transfer Learning Backbone
    + Custom Classification Head (5 classes)
    + K-Fold Cross Validation
    + Early Stopping
    + Model Checkpointing
    + Learning Rate Scheduling
```

---

## 10. Inference & Submission

```python
# Prediction pipeline
def predict(model, data_loader, device="cpu"):
    all_probs = torch.tensor([]).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)
            probs = torch.nn.functional.softmax(output, dim=1)
            all_probs = torch.cat((all_probs, probs), dim=0)

    return all_probs


```

---


---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-GPU_Accelerated-76B900?style=flat-square&logo=nvidia)