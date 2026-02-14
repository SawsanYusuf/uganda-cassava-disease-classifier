#  Crop Diseases in Uganda: Deep Learning for Cassava Leaf Classification


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


### 2. Transfer Learning with ResNet50

To improve performance, we implemented transfer learning using:

```python
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT
)
````

Instead of training from scratch:

* Pretrained ImageNet weights were used
* The final classifier layer was replaced
* K-Fold Cross Validation was applied to ensure stable performance

### K-Fold Validation Results

The model achieved:

* **~60–62% validation accuracy across folds**

This confirmed:

* Performance was consistent
* Improvement was not due to a single lucky split
* Transfer learning significantly outperformed the baseline CNN

The accuracy curve plateaued, indicating that learning stabilized and overfitting was controlled compared to the baseline model.


### 3. Final Training with Callbacks

After selecting the best-performing configuration, the model was retrained using:

* **StepLR Scheduler** (learning rate decreases every 4 epochs)
* **Model Checkpointing** (saving best model based on validation loss)

During training, we logged:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy
* Learning Rate

### Learning Curve Analysis

The learning curve shows:

* Validation loss decreases initially and stabilizes
* Training accuracy slightly higher than validation accuracy (expected behavior)
* Learning rate decreases every 4 epochs following StepLR schedule

This indicates stable convergence without severe overfitting.


## Model Evaluation

### Confusion Matrix Analysis

The final confusion matrix demonstrates:

* Most predictions lie on the diagonal
* The model performs well overall
* Some confusion exists between:

  * Bacterial Blight and Brown Streak (visual similarity)
  * Bacterial Blight and Healthy samples (possible feature overlap)

Despite these challenges, the model achieved meaningful classification capability for agricultural diagnostics.


## Technologies Used

* **Language:** Python
* **Framework:** PyTorch
* **Deep Learning Model:** ResNet50 (Transfer Learning)
* **Validation Strategy:** K-Fold Cross Validation
* **Optimization:** Adam Optimizer + StepLR Scheduler
* **Evaluation Tools:** Confusion Matrix, Learning Curves
* **Environment:** Jupyter Notebook


## Real-World Impact

This project demonstrates how deep learning can support:

* Early disease detection in agriculture
* Reduction of crop loss
* Data-driven farming practices
* Scalable AI-powered plant diagnostics

With further refinement and deployment, this system could be integrated into mobile-based diagnostic tools for farmers in low-resource environments.


## Future Improvements

* Apply stronger data augmentation techniques
* Fine-tune deeper layers of ResNet
* Experiment with weighted loss functions instead of undersampling
* Evaluate more advanced architectures (EfficientNet, Vision Transformers)
* Retrain final model on full balanced dataset
* Develop deployment pipeline for real-world agricultural use


## Author

**Sawsan Yousef**

Data Scientist | Machine Learning | Computer Vision
