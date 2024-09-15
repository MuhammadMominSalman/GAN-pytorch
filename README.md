
## Table of Contents

- [Description](#Description)
- [Requirements](#requirements)
- [GAN Architectures Implementation](#GANArchitecturesImplementation)
- [Usage](#usage)
- [Training](#training)

## Description

Generative Adversarial Networks (GANs) are a class of neural networks used to generate new data samples that resemble a given dataset. GANs consist of two neural networks, a generator and a discriminator, which are trained in an adversarial setup. The generator learns to produce fake data, and the discriminator learns to distinguish between real and fake data. Over time, the generator becomes better at creating realistic data, and the discriminator becomes better at distinguishing fake from real data.

In this project, we implement three types of GANs.

# GAN Architectures Implementation

This repository contains implementations of different Generative Adversarial Networks (GAN) architectures for generating images. The architectures implemented include:

1. **Simle GAN** - The basic GAN model with fully connected layers.
2. **Deep Convolutional GAN (DCGAN)** - A GAN using convolutional layers for higher quality image generation.
3. **Conditional GAN (CGAN)** - A GAN that generates class-specific images by conditioning on labels.


## Requirements

The following Python libraries are required to run the code:

- `torch`: For building and training neural networks.
- `torchvision`: For loading and preprocessing the MNIST dataset.
- `matplotlib`: For visualizing generated images.
- `tensorboard`: For visualizing generated images.

You can install the dependencies using the following command:

```bash
pip install torch torchvision matplotlib
```

## Usage

```bash
git clone https://github.com/yourusername/gan-mnist.git
cd gan-mnist
```

Install the required packages:

```bash
pip install -r requirements.txt
```

For MNIST dataset:
```bash
python train_MNIST.py
```
