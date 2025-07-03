# Deep Learning Applications: Laboratory #1

This notebook is a hands-on introduction to deep learning for visual recognition, focusing on Multilayer Perceptrons (MLPs), Residual Networks, and Convolutional Neural Networks (CNNs). The exercises are designed to help you understand the impact of network depth, residual connections, and transfer learning, using PyTorch and popular datasets like MNIST and CIFAR-10/100.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Lab Sections](#lab-sections)
  - [1. MLPs and Residual Connections](#1-mlps-and-residual-connections)
  - [2. CNNs and Deeper Architectures](#2-cnns-and-deeper-architectures)
  - [3. Advanced Exercises](#3-advanced-exercises)
    - [3.1 Fine-tuning a Pre-trained Model](#31-fine-tuning-a-pre-trained-model)
- [Dependencies](#dependencies)
- [Usage Tips](#usage-tips)
- [References](#references)

---

## Overview

- **Goal:** Explore the effects of network depth, residual connections, and transfer learning in deep learning models for image classification.
- **Datasets:** MNIST, CIFAR-10, CIFAR-100.
- **Skills:** Model implementation, training pipelines, experiment tracking, feature extraction, fine-tuning, knowledge distillation, and model explainability.

---


## Lab Sections

### 1. MLPs and Residual Connections

- Implement and train simple MLPs on MNIST.
- Experiment with deeper MLPs and observe the effect on training and validation performance.
- Implement residual connections in MLPs and compare their trainability and accuracy to standard MLPs.

### 2. CNNs and Deeper Architectures

- Implement flexible CNN architectures for CIFAR-10.
- Compare shallow and deep CNNs, with and without residual connections.
- Use experiment tracking tools (e.g., Weights & Biases) for systematic comparison.

### 3. Advanced Exercises

Choose at least one of the following advanced exercises:

#### 3.1 Fine-tuning a Pre-trained Model

- Use a CNN trained on CIFAR-10 as a feature extractor for CIFAR-100.
- Train a classical classifier (e.g., SVM, KNN) on extracted features for CIFAR-100.
- Fine-tune the CNN on CIFAR-100 and compare performance with the classical baseline.


---

## Dependencies

- Python 3.7+
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- wandb (Weights & Biases, for experiment tracking)
- google.colab (if running on Colab)

---

## Usage Tips

- **Reproducibility:** Set random seeds for torch and numpy.
- **Experiment Tracking:** Use Weights & Biases or TensorBoard for logging metrics and comparing runs.
- **Visualization:** Plot loss and accuracy curves for all experiments.
- **Modularity:** Abstract your model, training, and evaluation code for easy experimentation.
- **Documentation:** Clearly document all decisions, results, and conclusions as required by the lab.

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision.models.resnet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [Deep Residual Learning for Image Recognition (He et al., 2016)](https://arxiv.org/abs/1512.03385)
- [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [Class Activation Maps (Zhou et al., 2015)](http://cnnlocalization.csail.mit.edu/)
- [Weights & Biases](https://wandb.ai/site)

---

