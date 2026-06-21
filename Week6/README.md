# Image Denoising using Autoencoder on MNIST

## Overview

This project implements a **Denoising Autoencoder (DAE)** using PyTorch to remove artificial noise from handwritten digit images in the MNIST dataset.

The model is trained to reconstruct clean images from noisy inputs. By learning meaningful latent representations of handwritten digits, the autoencoder can effectively recover the original image while removing noise.

This project was developed as part of the **Week 6 Deep Learning Assessment**.

---

## Problem Statement

Image noise is a common issue in computer vision applications. The objective of this project is to train a neural network that can:

* Accept a noisy image as input
* Learn important features of handwritten digits
* Reconstruct a clean version of the image
* Preserve digit structure while removing unwanted noise

---

## Dataset

### MNIST Dataset

The MNIST dataset contains:

* 60,000 training images
* 10,000 testing images
* Grayscale handwritten digits (0–9)
* Image size: 28 × 28 pixels

Dataset is automatically downloaded using PyTorch's torchvision library.

---

## Project Workflow

### 1. Data Loading

The MNIST dataset is loaded using:

```python
torchvision.datasets.MNIST()
```

Images are converted into tensors and normalized to the range:

```text
[0, 1]
```

---

### 2. Noise Generation

Artificial Gaussian noise is added to clean images:

```python
noise = torch.randn_like(images) * 0.5
```

Noisy images are clipped to remain within the valid pixel range:

```python
torch.clamp(noisy_images, 0., 1.)
```

This creates the training pairs:

```text
Input  → Noisy Image
Target → Clean Image
```

---

### 3. Denoising Autoencoder Architecture

#### Encoder

The encoder compresses the image into a lower-dimensional representation.

```text
784 → 128 → 64
```

Layers:

* Linear(784,128)
* ReLU
* Linear(128,64)
* ReLU

---

#### Decoder

The decoder reconstructs the image from the compressed representation.

```text
64 → 128 → 784
```

Layers:

* Linear(64,128)
* ReLU
* Linear(128,784)
* Sigmoid

---

### Architecture Diagram

```text
Noisy Image (28×28)
          │
          ▼
       Flatten
          │
          ▼
      Encoder
    784 → 128
          │
          ▼
      128 → 64
          │
      Latent Space
          │
          ▼
      Decoder
      64 → 128
          │
          ▼
     128 → 784
          │
          ▼
 Reconstructed Image
```

---

## Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Optimizer     | Adam                     |
| Learning Rate | 0.001                    |
| Loss Function | Mean Squared Error (MSE) |
| Batch Size    | 128                      |
| Epochs        | 10                       |
| Framework     | PyTorch                  |

---

## Loss Function

The model uses Mean Squared Error (MSE) loss:

```python
criterion = nn.MSELoss()
```

MSE measures the difference between:

```text
Reconstructed Image
and
Original Clean Image
```

Lower loss indicates better reconstruction quality.

---

## Results

The trained model successfully learns to remove a significant amount of noise from handwritten digit images.

### Visualization

The project compares:

1. Original Image
2. Noisy Image
3. Reconstructed (Denoised) Image

Example output:

```text
Original Digit
      ↓
Added Noise
      ↓
Denoising Autoencoder
      ↓
Reconstructed Digit
```

The reconstructed images closely resemble the original handwritten digits while removing most of the artificial noise.

---

## Observations

* The autoencoder successfully learned the structure of handwritten digits.
* Gaussian noise was effectively reduced in reconstructed images.
* Major digit shapes were preserved after denoising.
* Training loss decreased consistently across epochs.
* Reconstructed images were smoother than noisy inputs.
* Some fine pixel-level details were slightly blurred due to compression in the latent space.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Google Colab / Jupyter Notebook

---

## Conclusion

A Denoising Autoencoder was successfully implemented using the MNIST dataset. The model learned to reconstruct clean handwritten digit images from noisy inputs and demonstrated effective image denoising capabilities. This project highlights the ability of autoencoders to perform image restoration tasks and serves as a foundation for more advanced image reconstruction techniques.
