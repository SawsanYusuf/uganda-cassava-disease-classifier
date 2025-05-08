# Cassava Disease Classification

This project focuses on classifying diseases affecting cassava plants, a crucial food source in Uganda and other regions globally. Imagine monitoring the health of your cassava farm by inspecting leaves for signs of disease. While seemingly straightforward, manual disease identification can be challenging, as normal leaf variations can resemble symptoms of serious infections. Accurate disease classification is vital as it dictates the appropriate treatment.

This is where the power of machine learning comes into play. The project aims to develop a computer vision model capable of analyzing images of cassava leaves and automatically determining if they are healthy or affected by a disease. For diseased plants, the model will further classify the specific type of ailment.

## Dataset Overview

The dataset comprises images of cassava leaves, organized into training and testing sets. The training set is further divided into five distinct folders, each representing a different class:

1.  **Healthy Cassava:** Images of healthy cassava leaves.
2.  **Bacterial Blight:** Images showing brown spots indicative of Bacterial Blight.
3.  **Brown Streak:** Images displaying the characteristic streaks of discoloration associated with Brown Streak disease.
4.  **Green Mottle:** Images exhibiting the symptoms of Green Mottle disease.
5.  **Mosaic Disease:** Images showing the mosaic-like pattern on leaves caused by Mosaic Disease.

Each image is labeled implicitly based on the folder it resides in. For instance, all images within the "Bacterial Blight" folder are labeled accordingly.

**Sample Images:**

* **Bacterial Blight:** Brown spots on the leaves.
* **Brown Streak:** Visible streaks of discoloration.
* **Healthy Cassava:** Leaves that may appear normal to the untrained eye.

The primary goal is to automate the classification process, which would be impractical to perform manually on a large scale. By building a robust computer vision model, we can efficiently and accurately categorize these images.

## Data Preparation

The cassava leaf images are loaded and preprocessed using PyTorch Tensors to prepare them for training deep learning models. The essential transformations applied include:

* **Converting to RGB:** Ensuring all images have three color channels.
* **Resizing:** Standardizing the size of all images to 224x224 pixels.
* **Converting to Tensors:** Transforming the images into PyTorch Tensor objects.
* **Normalization:** Scaling pixel values to have a mean of 0 and a standard deviation of 1 for each color channel. This helps in stabilizing and accelerating the training process.

A custom class was initially used to handle the conversion to RGB before applying the standard `torchvision.transforms` for resizing, tensor conversion, and normalization.

## Initial Model Training and Overfitting

The project initially involved building and training a Convolutional Neural Network (CNN) from scratch. This process included defining the network architecture, selecting an optimizer, and training the model on the prepared dataset. However, significant overfitting was observed during this phase.

The symptoms of overfitting were evident in:

* A substantial gap between the training accuracy (high) and the validation accuracy (plateaued). This indicated that the model was memorizing the training data instead of learning generalizable features.
* A confusion matrix revealing high misclassification rates across different disease categories, further confirming the model's poor generalization ability.

## Leveraging Transfer Learning with ResNet-50

To address the overfitting issue and improve model performance, the project transitioned to using **transfer learning**. This technique involves leveraging pre-trained models that have been trained on massive datasets (like ImageNet) and adapting them to our specific task.

We utilized the **ResNet-50** architecture, a deep convolutional neural network available through TorchVision. By loading the pre-trained weights of ResNet-50, we could benefit from the features learned by the model on a vast number of images.

The adaptation process involved:

* Loading the ResNet-50 architecture using `torchvision.models.resnet50()`.
* Fetching the pre-trained weights using `torchvision.models.ResNet50_Weights`.
* Replacing the original classification layer (designed for 1000 ImageNet classes) with a new, untrained fully connected (FC) layer tailored to our 5 cassava disease classes. This new classifier consisted of:
    1.  A Linear layer mapping 2048 inputs to 256 neurons.
    2.  A ReLU activation function for introducing non-linearity.
    3.  A Dropout layer to mitigate overfitting.
    4.  A final Linear layer mapping 256 neurons to the 5 output classes.

The core idea was to freeze the weights of the pre-trained ResNet-50 layers and only train the newly added classification layers. This allows us to fine-tune the model for our specific task without needing a large dataset or extensive training time.

## Implementing K-Fold Cross-Validation

To obtain a more reliable evaluation of the fine-tuned model's performance, **k-fold cross-validation** was implemented. Instead of a single 80/20 train-validation split, the dataset was divided into K equal-sized folds. The training process was repeated K times, with each fold serving as the validation set once, while the remaining K-1 folds were used for training. The final performance was the average of the results across all K iterations.

This approach provides a more robust estimate of the model's generalization ability by reducing the impact of any potentially biased single train-validation split. While it increases the training time, the more reliable evaluation is valuable.

The results of the k-fold cross-validation showed:

* Consistent decrease in training loss across folds.
* More variation in validation loss across folds.
* Validation accuracy averaging around 61%, with training loss and validation loss being relatively close (around 0.8), suggesting minimal overfitting.

A confusion matrix generated from the final fold's validation set revealed some misclassifications, particularly between Bacterial Blight and Brown Streak, likely due to visual similarities. Misclassifications between Bacterial Blight and healthy samples also suggested potential dataset imbalance or feature extraction challenges.

## Enhancing Training with Callbacks

To further improve the training process and model performance, **callbacks** were introduced. Callbacks are special functions that can be executed at different stages of the training process to modify its behavior dynamically. Three key callbacks were implemented:

1.  **Learning Rate Scheduling:** Adjusting the learning rate during training. Specifically, a `StepLR` scheduler was used to decrease the learning rate by a factor of 0.1 every 4 epochs. This helps the optimization process converge more effectively.
2.  **Checkpointing:** Saving the model's state (weights) whenever the validation loss improves. This ensures that the best-performing model during training is preserved.
3.  **Early Stopping:** Monitoring the validation loss and stopping the training process if it doesn't improve for a specified number of epochs (patience). This prevents overfitting by stopping training when the model starts to generalize poorly on unseen data.

By implementing these callbacks, the training process became more efficient and robust:

* The learning rate was gradually reduced, aiding convergence.
* Overfitting was mitigated by stopping training when validation performance plateaued.
* The best model based on validation performance was automatically saved.

The results after training with callbacks showed:

* Validation loss not improving significantly after the initial epochs, indicating early stabilization.
* Training accuracy being slightly higher than validation accuracy, suggesting good generalization without significant overfitting.
* The learning rate decreasing as scheduled.

The final confusion matrix indicated good overall performance, with most predictions falling on the diagonal, signifying correct classifications.

## Potential Future Improvements

Despite the promising results, further improvements could be explored:

* **Utilizing Larger Pre-trained Models:** Experimenting with more complex pre-trained architectures like EfficientNet or Vision Transformers.
* **Applying Data Augmentation:** Increasing the diversity of the training data through transformations like rotations, flips, and zooms to reduce overfitting.
* **Fine-tuning More Layers:** Gradually unfreezing and fine-tuning more layers of the pre-trained ResNet-50 model.
* **Training on the Entire Dataset:** After k-fold validation, training the final model on the entire dataset using the best hyperparameters found during cross-validation.

This documentation provides an overview of the cassava disease classification project, outlining the problem, dataset, methodologies employed (including transfer learning and callbacks), and potential avenues for future enhancements.
