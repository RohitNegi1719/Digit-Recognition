# Neural Network for MNIST Digit Classification

This Python script implements a simple neural network from scratch using NumPy to classify digits from the MNIST dataset.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.x
- NumPy
- mlxtend (for loading the MNIST dataset)

## Getting Started

1. Clone this repository to your local machine or download the script.
2. Make sure you have the MNIST dataset files (`train-images.idx3-ubyte` and `train-labels.idx1-ubyte`) available locally. You can download them from [here](http://yann.lecun.com/exdb/mnist/).
3. Update the `images_path` and `labels_path` variables in the script with the correct paths to your local MNIST dataset files.
4. Run the script `neural_network_mnist.py`.

## Overview

- `neural_network_mnist.py`: The main Python script containing the implementation of the neural network, training, and testing procedures.
- `neural_network_model.npz`: The saved model file containing the trained weights and biases of the neural network.

## Usage

1. Run the script `neural_network_mnist.py`.
2. The script will load the MNIST dataset, train the neural network, save the trained model, and test its accuracy on a separate test dataset.
3. The accuracy of the model on the test dataset will be printed to the console.

## Author

Rohit Negi


