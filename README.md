#  Crop Diseases in Uganda: Deep Learning for Cassava Leaf Classification

![](https://github.com/SawsanYusuf/Cassava-Disease-Image-Classification/blob/main/Cassava%20Plant.jpg)

## Rapid and Accurate Diagnostic Aid for Cassava Health

This project implements a deep learning solution for the automated classification of cassava leaf images, aiming to identify common diseases affecting this crucial crop in Uganda. By leveraging Convolutional Neural Networks (CNNs), the model can distinguish between healthy leaves and various disease states, providing a vital tool for farmers to manage crop health, prevent widespread outbreaks, and ensure food security.

## Project Highlights

* **Multi-Class Classification:** Differentiates between five distinct cassava leaf conditions: 'cassava-healthy', 'cassava-mosaic-disease-cmd', 'cassava-brown-streak-disease-cbsd', 'cassava-green-mottle-cgm', and 'cassava-bacterial-blight-cbb'.
* **Transfer Learning with ResNet-18:** Utilizes a powerful pre-trained CNN architecture (ResNet-18) as a feature extractor, adapting it for high-accuracy performance on cassava leaf images.
* **Robust Data Handling:** Implements meticulous data preprocessing, including image standardization and normalization, crucial for deep learning models.
* **Class Imbalance Mitigation:** Addresses imbalanced class distribution in the dataset through an effective undersampling strategy during training.
* **Comprehensive Evaluation:** Provides detailed performance metrics (accuracy, precision, recall, F1-score) and visual error analysis on an unseen test set.

## Dataset

The model is trained and evaluated on a structured dataset of cassava leaf images. The images are categorized into the five classes mentioned above, representing different health states of the plant. The dataset was split into training, validation, and test sets (70%, 20%, 10% respectively) to ensure robust model development and unbiased evaluation.

## Methodology

Our approach involved a systematic deep learning pipeline built using PyTorch:

1.  **Environment Setup:** Python 3.11.0 and PyTorch 2.2.2+cu121 were configured, leveraging CUDA for GPU acceleration.
2.  **Data Preprocessing & Augmentation:**
    * Images were resized to 224x224 pixels and converted to RGB format.
    * Global mean and standard deviation were calculated from the dataset for image normalization.
    * An **undersampling strategy** was applied to the training set to ensure a balanced distribution of classes, preventing the model from being biased towards majority classes.
3.  **Model Architecture:**
    * A pre-trained **ResNet-18** model was used for transfer learning.
    * The convolutional layers (feature extractor) were frozen, and a new custom classification head was added, tailored to output predictions for the five cassava leaf classes.
4.  **Model Training:**
    * The model was trained using the **Adam optimizer** and `nn.CrossEntropyLoss`.
    * A **`StepLR` learning rate scheduler** was implemented to gradually reduce the learning rate during training, aiding in better convergence.
    * Training was monitored on the validation set, and the best-performing model weights (based on validation accuracy) were saved.
5.  **Model Evaluation:**
    * The final model was rigorously evaluated on a completely unseen test set.
    * Performance was assessed using overall test accuracy, per-class precision, recall, and F1-score.
    * A confusion matrix was generated to visualize classification results, and a dedicated error analysis provided insights into misclassified images.

## Key Results

The ResNet-18 model demonstrated **exceptional performance** on the unseen test set for cassava disease classification:

* **Test Loss:** `0.0526`
* **Test Accuracy:** `0.9859` (98.59%)

**Detailed Classification Report (on Test Set):**

| Class                               | Precision | Recall | F1-Score | Support |
| :---------------------------------- | :-------- | :----- | :------- | :------ |
| `cassava-bacterial-blight-cbb`      | 0.99      | 0.99   | 0.99     | 152     |
| `cassava-brown-streak-disease-cbsd` | 0.99      | 0.99   | 0.99     | 360     |
| `cassava-green-mottle-cgm`          | 0.98      | 0.98   | 0.98     | 313     |
| `cassava-healthy`                   | 0.98      | 0.97   | 0.97     | 286     |
| `cassava-mosaic-disease-cmd`        | 0.99      | 0.99   | 0.99     | 379     |
| **Macro Avg** | 0.99      | 0.98   | 0.98     | 1490    |
| **Weighted Avg** | 0.99      | 0.99   | 0.99     | 1490    |

These results highlight the model's consistent high performance across all classes, indicating its reliability in accurately identifying both healthy and various diseased cassava leaves.

## Visualizations

The project notebook includes key visualizations to understand model behavior:

* **Training & Validation Loss/Accuracy Plots:** Illustrate the model's learning progression over epochs.
* **Class Distribution Plots:** Show the distribution of classes before and after undersampling, confirming successful data balancing.
* **Confusion Matrix:** Provides a detailed breakdown of correct and incorrect predictions on the test set.
* **Error Analysis Plots:** Visualizes examples of correctly and incorrectly classified images, aiding in understanding model strengths and areas for potential improvement.

## Future Work

* **Deployment:** Develop a user-friendly application (e.g., mobile app) for real-time disease diagnosis in the field.
* **Model Optimization:** Explore model quantization or pruning for efficient deployment on edge devices.
* **Expanded Datasets:** Test the model on new, diverse datasets from various regions and conditions to ensure broader applicability.
* **Explainable AI (XAI):** Implement Grad-CAM to visualize model attention, enhancing trust and providing insights to agricultural experts.
* **Multi-Crop/Multi-Disease Scope:** Extend the model to classify diseases in other crops common to the region.

## Contributing

Contributions, issues, and feature requests are welcome! 

## Author
**Sawsan Yousef** 

*Data Scientist | Predictive Modeling | Computer Vision*
