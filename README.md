# Cassava Leaf Disease Classification Using Transfer Learning (ResNet50)

## Project Overview

Cassava is a critical food crop in many African countries, including Uganda. However, viral and bacterial diseases significantly reduce crop yield and threaten food security.

This project develops a deep learning-based image classification system to detect cassava leaf diseases using **Transfer Learning with ResNet50**. The objective is to build a robust and generalizable model capable of accurately identifying plant health conditions from leaf images.

The project follows a structured experimentation process:
- Building a baseline CNN from scratch
- Applying Transfer Learning with K-Fold Cross Validation
- Refining the final model using callbacks and checkpointing



## Dataset Description

The dataset consists of approximately **21,000 cassava leaf images** divided into five classes:

- Cassava Bacterial Blight (CBB)
- Cassava Brown Streak Disease (CBSD)
- Cassava Mosaic Disease (CMD)
- Cassava Green Mottle (CGM)
- Healthy Leaves

### Class Imbalance Handling

The dataset was originally imbalanced, with significantly more samples belonging to common diseases compared to rarer but critical target diseases.

To ensure fair learning across all classes, **Random Undersampling** was applied.  
After balancing, the dataset contained:

- **1,523 images per class**
- **Total balanced dataset: 7,615 images**

This approach ensured equal representation and prevented bias toward dominant classes.


## Data Preprocessing

The following preprocessing steps were applied:

- Resized all images to **224 × 224**
- Converted all images to **RGB format**
- Applied **Normalization**
- Converted images into **PyTorch tensors**

These steps ensured compatibility with pretrained ResNet architecture and stabilized training.


## Model Development

### 1. Baseline CNN (From Scratch)

A custom CNN model was first built using standard convolutional layers.

- Validation Accuracy: ~40%
- Observation: Clear gap between training and validation accuracy
- Conclusion: The model suffered from **overfitting** and poor generalization

---

### 2. Transfer Learning with ResNet50

To improve performance, we implemented transfer learning using:

```python
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT
)
```

