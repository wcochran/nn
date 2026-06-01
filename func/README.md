# Function Approximator

Small educational C++ project that trains a neural network from scratch to fit
scalar functions on an interval. It uses Eigen for matrix and vector math, but
does not use a machine-learning framework.

The network is a simple regression model:

- input size: 1
- hidden layer 1: configurable, default 16 neurons
- hidden layer 2: configurable, default 16 neurons
- output size: 1
- hidden activation: configurable, default ReLU
- output activation: linear
- loss: mean squared error
- optimizer: mini-batch stochastic gradient descent

The implementation keeps the forward-pass cache and backpropagation equations
explicit in `neural_network.cpp` so the math is easy to inspect.

## Requirements

- C++17 compiler
- CMake
- Eigen3
- gnuplot, optional, for plotting `predictions.csv`

On Ubuntu/Debian:

```bash
sudo apt-get install -y cmake g++ libeigen3-dev gnuplot
```

## Build

Use a Release build for training speed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Train

Run with defaults:

```bash
./build/function_approximator
```

The default target is `erf` on `[-3, 3]` with 5000 epochs, batch size 32,
learning rate 0.01, hidden size 16, and 1024 sampled points per epoch.

Arguments:

```text
./build/function_approximator [function] [xmin] [xmax] [epochs] [hidden_size] [batch_size] [learning_rate] [samples_per_epoch] [activation]
```

Available functions:

```text
erf cos sin sin5 gaussian square tanh
```

Available hidden activations:

```text
relu tanh sigmoid
```

Example:

```bash
./build/function_approximator erf -3 3 5000 32 32 0.005 4096
```

Example using tanh hidden activations:

```bash
./build/function_approximator erf -3 3 5000 32 32 0.005 4096 tanh
```

Training writes `predictions.csv` with:

```text
x,true_y,predicted_y,error
```

## Plot

After training:

```bash
gnuplot plot_predictions.gp
```

This writes `predictions.png`, overlaying the true and predicted values.

## Training Terms

An epoch is one training cycle. In this project, an epoch means sampling
`samples_per_epoch` random x values from the interval and training on them.
Because samples are drawn randomly, each epoch sees fresh points rather than a
fixed dataset.

A batch is the group of examples used before one parameter update. With
`samples_per_epoch = 4096` and `batch_size = 32`, one epoch has:

```text
4096 / 32 = 128 batches
```

An update is one change to the weights and biases. Gradients are accumulated
over one batch, then SGD updates `W1`, `b1`, `W2`, `b2`, `W3`, and `b3`.
Therefore:

```text
1 batch = 1 update
```

For the example command above:

```text
5000 epochs
128 updates per epoch
640000 total updates
20480000 total sampled training examples
```
