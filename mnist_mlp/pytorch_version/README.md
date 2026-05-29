# MNIST PyTorch Example

This project trains a simple fully connected neural network on the MNIST dataset using PyTorch and `torchvision`.

## Files

- `train_mnist.py`: single runnable training script
- `requirements.txt`: Python dependencies for the script

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python train_mnist.py
```

The script will:

- download MNIST automatically into `../data/MNIST/raw` shared with the C++ implementation
- train for 5 epochs
- print training loss and test accuracy after each epoch

## Notes

- The script runs on CPU by default and will use CUDA automatically if available.
- The model is a simple fully connected network with no convolutional layers.
