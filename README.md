# AI Deep Facial Recognition

## Summary

This repository provides an implementation of a facial recognition model inspired by the paper "Siamese Neural Networks for One-Shot Image Recognition." The project addresses the challenge of one-shot learning, where models must accurately recognize and classify new categories using just a single example per class. This capability is crucial for applications where gathering extensive datasets is impractical. The method uses Siamese neural networks, which can effectively learn a similarity measure between pairs of inputs, enhanced by a convolutional architecture to generalize across new categories efficiently.

## Table of Contents

1. [Introduction](#introduction)
2. [Concepts Involved](#concepts-involved)
   - [One-Shot Learning](#1-one-shot-learning)
   - [Siamese Neural Networks](#2-siamese-neural-networks)
   - [Convolutional Neural Networks (CNNs)](#3-convolutional-neural-networks-cnns)
3. [Getting Started](#getting-started)
4. [GPU Configuration](#gpu-configuration)
5. [Running the Code](#running-the-code)
6. [Deactivation and Cleanup](#deactivation-and-cleanup)
7. [References](#references)

## Introduction

This repository implements a Siamese neural network for one-shot learning based on the principles presented in the relevant research paper. The focus is on achieving high accuracy in image classification with minimal data, leveraging the strength of Siamese networks to compare and learn from pairs of images.

## Concepts Involved

### 1. One-Shot Learning

- **The Challenge:** Train a model to recognize categories from a single example.
- **Why It’s Hard:** Models typically require large datasets for accuracy; one-shot learning mimics human capability of learning from few examples.
- **Relevance:** Enables the model to resolve image classification tasks with minimal data.

### 2. Siamese Neural Networks

- **Twin Networks:** Consist of two identical subnetworks to process input images.
- **Comparison Mechanism:** Outputs are compared using a distance metric to measure similarity.
- **Application:** Useful for ranking similar images, central to one-shot tasks.

### 3. Convolutional Neural Networks (CNNs)

- **Feature Extraction:** Utilizes convolutional layers to detect image features like edges and textures.
- **Robustness:** Excel at learning features invariant to image changes (such as rotations).
- **Significance:** Essential in Siamese networks for extracting relevant features for comparison.

## Getting Started

### Install Visual Studio Code
- Download and install [Visual Studio Code](https://code.visualstudio.com/).

### Set Up Python Environment
- Ensure Python is installed. [Download Python](https://www.python.org/downloads/).
- Install `virtualenv`:
  ```bash
  pip install virtualenv
  ```

### Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### Create and Activate a Virtual Environment
```bash
python -m venv env
```
- **Activate:**
  - Windows:
    ```bash
    .\env\Scripts\activate
    ```
  - MacOS/Linux:
    ```bash
    source env/bin/activate
    ```

## GPU Configuration

1. **Install NVIDIA Drivers:**
   - Download from the [NVIDIA driver page](https://www.nvidia.com/Download/index.aspx).

2. **Install CUDA Toolkit and cuDNN:**
   - Download from [CUDA Toolkit](https://developer.nvidia.com/cuda-11.4.1-download-archive) and [cuDNN](https://developer.nvidia.com/cudnn).

3. **Set Environment Variables (Windows):**
   - Add CUDA `bin` and `lib` directories to `Path`.

4. **Verify TensorFlow GPU Support:**
   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
   ```

## Running the Code

### Install Required Packages
- Ensure `requirements.txt` includes necessary dependencies, e.g.:
  ```
  tensorflow==2.4.1
  opencv-python
  matplotlib
  ```
- Install them:
  ```bash
  pip install -r requirements.txt
  ```

### Open Project in VS Code
- Load your project folder and select the virtual environment:
  - `Ctrl` + `Shift` + `P` → `Python: Select Interpreter`.

### Run the Script
- Ensure that all files (like `lfw.tgz`) are present.
- Run using the VS Code Run button or press `F5`.

## Deactivation and Cleanup

- Once finished, deactivate the virtual environment:
  ```bash
  deactivate
  ```

## References

- **Original Paper:** ["Siamese Neural Networks for One-Shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by Koch et al.
- **TensorFlow Documentation:** [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- **VS Code Documentation:** [Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)