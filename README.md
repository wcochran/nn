# Neural Network Projects

This repository is a workspace for small neural-network projects and experiments.

## Projects

### `mnist_mlp/`

An educational MNIST digit classifier implemented from scratch in C++ using Eigen.
The goal is to make the math visible rather than hide it behind a machine-learning
framework.

The C++ implementation includes:

- a `784 -> 16 -> 16 -> 10` fully connected network
- ReLU hidden activations
- softmax output
- cross-entropy loss
- explicit forward propagation
- explicit backpropagation
- mini-batch stochastic gradient descent
- an IDX-format MNIST loader

The project also includes `mnist_mlp/pytorch_version/`, a compact PyTorch baseline
for comparison. It uses a wider `784 -> 128 -> 10` model, Adam, PyTorch autograd,
and `nn.CrossEntropyLoss`.

## Data

MNIST data is shared by the C++ and PyTorch versions at:

```text
mnist_mlp/data/MNIST/raw/
```

The data files are intentionally ignored by Git. To fetch them:

```bash
cd mnist_mlp
./fetch_mnist.sh
```

## Build and Run the C++ Version

```bash
cd mnist_mlp
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
./build-release/mnist_mlp
```

## Run the PyTorch Version

```bash
cd mnist_mlp/pytorch_version
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python train_mnist.py
```

## Documentation

The LaTeX source for the educational write-up is here:

```text
mnist_mlp/doc/mnist_mlp.tex
```

A generated PDF is also included:

```text
mnist_mlp/doc/mnist_mlp.pdf
```
