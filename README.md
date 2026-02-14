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

These preprocessing steps ensured compatibility with pretrained CNN architectures such as ResNet50 and improved convergence during training.


## Model Development & Evaluation

### Baseline Model – Custom CNN

We first built a Convolutional Neural Network (CNN) from scratch using standard convolutional and fully connected layers.

The model achieved approximately **40% validation accuracy**.

During training, a significant gap was observed between training and validation accuracy. While training accuracy continued to increase, validation accuracy plateaued early. This indicates **overfitting**, meaning the model memorized training samples instead of learning generalizable features.

This result established a clear baseline and motivated the use of Transfer Learning.


### Transfer Learning with ResNet50 + K-Fold Validation

To improve performance, we adopted a pretrained **ResNet50** model from `torchvision`:

```python
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT
)
```

To ensure the performance was not due to a lucky split, we applied **K-Fold Cross Validation**.

This allowed us to:

* Train and validate the model across multiple data splits
* Reduce variance caused by a single train-validation split
* Evaluate robustness across the entire dataset

The model achieved approximately **60% validation accuracy across folds**, representing a significant improvement over the custom CNN baseline.

At this stage, the Transfer Learning model was selected as the final architecture.


### Final Training with Callbacks

After selecting the best architecture, we retrained the model using training callbacks to further improve stability and performance.

The following techniques were applied:

* **StepLR Scheduler**
  The learning rate decreased every 4 epochs, allowing smoother convergence.

* **Checkpointing**
  The best model was saved whenever validation loss improved.

* **Learning Curve Monitoring**
  Training loss, validation loss, accuracy, and learning rate were logged at each epoch.

#### Learning Curve Interpretation

The learning curve shows how the model improves over time:

* Validation loss decreased initially and then stabilized.
* Training accuracy remained slightly higher than validation accuracy (expected behavior).
* No severe divergence between curves, indicating controlled overfitting.
* Learning rate decreased according to the StepLR schedule.

After training, the best-performing model (based on validation loss) was loaded for final evaluation.



### Confusion Matrix Analysis

The final confusion matrix showed strong diagonal dominance, indicating that most predictions were correctly classified.

However, some misclassifications were observed:

* Bacterial Blight and Brown Streak were frequently confused due to visual similarity.
* Some overlap between disease classes and healthy leaves suggests feature-level ambiguity or dataset imbalance.

Overall, the model demonstrates stable learning behavior and reasonable classification capability for a real-world agricultural dataset.


## Technologies Used

* **Language:** Python
* **Framework:** PyTorch
* **Deep Learning Model:** ResNet50 (Transfer Learning)
* **Validation Strategy:** K-Fold Cross Validation
* **Optimization:** Adam Optimizer + StepLR Scheduler
* **Evaluation Tools:** Confusion Matrix, Learning Curves
* **Environment:** Jupyter Notebook


## Real-World Impact

This project demonstrates how Deep Learning can support agricultural sustainability by:

* Enabling early disease detection in cassava crops

* Supporting farmers in remote regions with automated diagnosis tools

* Reducing crop loss and improving food security

* Providing a scalable foundation for mobile-based plant disease detection systems


With further optimization and deployment, this model could be integrated into mobile applications to assist farmers in real-time crop monitoring.


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
