# CIFAR-10 Image Classification using ANN and CNN

## Overview

This assignment implements image classification on the CIFAR-10 dataset using Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The objective is to compare the performance of different neural network architectures and training strategies for image classification.

# Dataset
The CIFAR-10 dataset consists of 60,000 color images belonging to 10 classes.

### Dataset Statistics
* Total Images: 60,000
* Training Images: 50,000
* Test Images: 10,000
* Image Size: 32 × 32 × 3
* Number of Classes: 10

### Classes
* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

## Project Objectives
* Load and explore the CIFAR-10 dataset
* Perform image preprocessing and normalization
* Build an Artificial Neural Network (ANN)
* Build a Convolutional Neural Network (CNN)
* Implement Batch Normalization
* Implement Data Augmentation
* Compare model performances
* Visualize results using confusion matrices

## Technologies Used
* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Models Implemented

### 1. Artificial Neural Network (ANN)
Architecture:
* Dense Layer (512 neurons)
* Dropout Layer
* Dense Layer (256 neurons)
* Dropout Layer
* Output Layer (Softmax)

### 2. Convolutional Neural Network (CNN)
Architecture:
* Conv2D (32 filters)
* MaxPooling
* Conv2D (64 filters)
* MaxPooling
* Conv2D (128 filters)
* Dense Layer
* Output Layer

### 3. CNN with Batch Normalization
* Batch Normalization after convolution layers

Benefits:
* Faster convergence
* Improved training stability
* Better generalization

### 4. CNN with Data Augmentation
Augmentation Techniques:
* Horizontal Flip
* Rotation
* Zoom

Benefits:
* Reduces overfitting
* Improves robustness
* Increases effective training data diversity

## Training Strategy

The models were trained using:
* Adam Optimizer
* Sparse Categorical Crossentropy Loss
* Validation Split
* Mini-batch Training

## Evaluation Metrics

The following metrics were used:
* Accuracy
* Loss
* Classification Report
* Confusion Matrix

## Results

The performance of the following models was compared:

1. ANN
2. CNN
3. CNN + Batch Normalization
4. CNN + Data Augmentation

The results show that CNN-based architectures significantly outperform ANN models because they preserve and learn spatial information from images.

## Key Observations

* ANN achieved lower accuracy due to loss of spatial information after flattening.
* CNN extracted meaningful visual features and achieved better performance.
* Batch Normalization improved training stability.
* Data Augmentation improved generalization and reduced overfitting.
* CNN-based models consistently outperformed the ANN model.
