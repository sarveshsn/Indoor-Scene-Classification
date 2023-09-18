# Indoor Scene Classification

![000022](https://github.com/sarveshsn/Indoor-Scene-Classification/assets/93898181/fc658f7f-0270-4304-94f1-477a05eb6722)


## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Models](#models)
- [Execution](#execution)

## Project Description

This project focuses on the classification of indoor scenes using Convolutional Neural Networks (CNNs). The goal is to classify indoor images into 10 different categories, including bathrooms, bedrooms, and more.

## Dataset

The dataset used for this project is named `data_indoor`, and it consists of three main subsets:
- **Train**: Contains a large number of labeled images for model training.
- **Validation**: Used to fine-tune model hyperparameters and prevent overfitting.
- **Test**: Reserved for evaluating the model's performance on unseen data.

### Classes

The dataset comprises images from 10 different indoor scene categories, making it a multi-class classification problem:

1. Bathroom
2. Bedroom
3. Kitchen
4. Living Room
5. Children's Room
6. Closet
7. Corridor
8. Dining Room
9. Garage
10. Stairs

## Models

### Model 1

We have deployed a CNN model with the following architecture:
- Input Layer: (64, 64, 3) shape
- Convolutional Layers with ReLU Activation
- Max-Pooling Layers
- Fully Connected Layers
- Output Layer with Softmax Activation

The model is trained using RMSprop optimizer with a learning rate of 0.0001.

### Model 2

For this model, we increased the number of filters in convolutional layers and introduced batch normalization and dropout for regularization. We also varied the kernel size in convolutional layers and used the Adam optimizer.

### Model 3 (Data Augmentation)

To prevent overfitting, we employed data augmentation techniques, including rotation, shifting, shearing, and flipping. This model is similar to Model 1 but with data augmentation applied during training.

## Execution

To execute this project, follow these steps:

1. Load the dataset.
2. Train and evaluate the models (Model 1, Model 2, or Model 3) using the provided code.
3. Experiment with different hyperparameters, architectures, and data augmentation techniques to improve model performance.

### Additional Tips for Improvement

- Increase the number of filters in convolutional layers.
- Use dropout regularization.
- Apply batch normalization.
- Vary the kernel size in convolutional layers.

### Evaluation and Metrics

After training your models, evaluate them on the test set and calculate various metrics, including accuracy, precision, recall, and F1 score, to assess model performance.

## Author 

- **Sarvesh Sairam Naik**

