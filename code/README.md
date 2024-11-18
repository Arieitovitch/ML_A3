# OrganAMNIST Classification with Multilayer Perceptrons (MLP) and Convolutional Neural Networks (CNN)

This project involves building a classification system for the OrganAMNIST dataset using Multilayer Perceptrons (MLPs) implemented from scratch, as well as exploring Convolutional Neural Networks (CNNs).

## Features
- **Custom MLP Implementation**:
  - Configurable number of hidden layers and activation functions.
  - Supports ReLU, Tanh, and Softmax activations.
  - Implements forward and backward propagation from scratch using NumPy.
- **Data Preparation**:
  - Processes the OrganAMNIST dataset for both MLP and CNN usage.
  - Provides normalized and flattened data for MLPs.
  - Prepares 2D image data for CNNs.

---

## Getting Started

### Prerequisites
This project uses a **Conda environment** for managing dependencies. The `environment.yml` file specifies all required packages.

---

### Setting Up the Environment

1. **Create and Activate the Environment**:
   Use the provided `environment.yml` file to create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate venv

2. **Running On The GPU**:
    The launch file will run all code in the run.py file so we will run everything from out of there
