# Variational Autoencoder (VAE) on MNIST Dataset

![GitHub Repo Stars](https://img.shields.io/github/stars/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation?style=social)
![GitHub Forks](https://img.shields.io/github/forks/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation?style=social)
![GitHub Issues](https://img.shields.io/github/issues/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation)
![GitHub Last Commit](https://img.shields.io/github/last-commit/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation)
![GitHub Contributors](https://img.shields.io/github/contributors/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation)
![GitHub Repo Size](https://img.shields.io/github/repo-size/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation)
![GitHub Language Count](https://img.shields.io/github/languages/count/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation)
![GitHub Top Language](https://img.shields.io/github/languages/top/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation)
![GitHub Watchers](https://img.shields.io/github/watchers/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation?style=social)
![Commit Activity](https://img.shields.io/github/commit-activity/m/arpsn123/Variational-Autoencoder-for-MNIST-Handwritten-Digit-Generation)
![Maintenance Status](https://img.shields.io/badge/Maintained-Yes-green)



## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Installation](#installation)
- [Training](#training)
- [Conclusion](#conclusion)
- [License](#license)

## Overview

This project implements a **Variational Autoencoder (VAE)** using the MNIST dataset, which consists of 70,000 handwritten digits (0-9) divided into 60,000 training images and 10,000 test images. The VAE is a powerful generative model that learns to represent high-dimensional data in a lower-dimensional latent space, enabling it to generate new data points that are similar to the input data. This project serves as an educational example for understanding the fundamentals of VAEs and their application in image generation tasks.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F20?style=flat&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003B57?style=flat&logo=matplotlib&logoColor=white)

## Features

- **Variational Autoencoder Architecture**: The model consists of an encoder that compresses input images into a latent space representation and a decoder that reconstructs images from this representation.
- **Training on MNIST Dataset**: The VAE is trained on the widely-used MNIST dataset, which is a standard benchmark for image classification and generation tasks.
- **Sample Generation**: The model can generate new handwritten digit images based on the learned latent distribution.
- **Visualization Tools**: Includes scripts for visualizing training progress and generated samples.
- **Configurable Training Parameters**: Users can easily adjust hyperparameters such as learning rate, batch size, and number of epochs.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**: Start by cloning the project repository to your local machine:

   ```bash
   git clone https://github.com/arpsn123/vae-mnist.git
   cd vae-mnist
   ```

2. **Set Up a Virtual Environment** (Optional but recommended): It is good practice to use a virtual environment to manage project dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Dependencies**: Install the necessary libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all required packages, including TensorFlow or PyTorch, NumPy, and Matplotlib, depending on the implementation.



## Training the VAE

To train the VAE model, execute the following command in your terminal:

```bash
python train.py
```

- This command will initiate the training process using the MNIST dataset. During training, the model learns to map input images to a latent space and back, optimizing the reconstruction of the images while minimizing the difference between the learned and prior distributions.
- The training logs will be generated to monitor progress, and model checkpoints will be saved periodically.

### Generating Samples

Once the model has been trained, you can generate new samples with the following command:

```bash
python generate.py
```

- This script utilizes the trained model to produce new images of handwritten digits by sampling from the learned latent space.
- Generated samples will be saved in a designated output directory, allowing you to visualize the results.

## Training

The model is trained with the following configurations:

- **Dataset**: MNIST
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Epochs**: 50
- **Loss Function**: A combination of reconstruction loss (mean squared error between input and output) and KL divergence (measuring how one probability distribution diverges from a second expected probability distribution).

### Training Process

1. **Data Preparation**: The MNIST dataset is loaded and preprocessed. Images are normalized to a range of [0, 1] for better training performance.
2. **Model Initialization**: The encoder and decoder components of the VAE are initialized.
3. **Training Loop**: For each epoch, the model is trained on batches of images, calculating the loss and updating the weights using backpropagation.
4. **Monitoring Performance**: The training process outputs loss metrics, allowing users to track how well the model is learning over time.



### Evaluation of Results

- The quality of generated images can be evaluated qualitatively by visual inspection and quantitatively using metrics such as Inception Score (IS) or Fréchet Inception Distance (FID) for more advanced analyses.
- Visualizations of reconstructed images from the test set can also be provided to assess the model's performance in reconstructing known inputs.

## Conclusion

This VAE implementation showcases the ability to generate new handwritten digits similar to those in the MNIST dataset, providing insights into generative modeling techniques. By understanding VAEs, users can explore further applications in various fields such as semi-supervised learning, anomaly detection, and more complex generative tasks.

This project can be extended by experimenting with different architectures, using larger datasets, or integrating advanced techniques like attention mechanisms for improved performance.

