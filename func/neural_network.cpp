#include "neural_network.h"

#include <cmath>
#include <stdexcept>

std::vector<std::string> availableActivationNames() {
    return {"relu", "tanh", "sigmoid"};
}

Activation activationFromName(const std::string& name) {
    if (name == "relu") {
        return Activation::Relu;
    }
    if (name == "tanh") {
        return Activation::Tanh;
    }
    if (name == "sigmoid") {
        return Activation::Sigmoid;
    }

    throw std::invalid_argument("unknown activation: " + name);
}

std::string activationName(Activation activation) {
    switch (activation) {
        case Activation::Relu:
            return "relu";
        case Activation::Tanh:
            return "tanh";
        case Activation::Sigmoid:
            return "sigmoid";
    }

    throw std::invalid_argument("unknown activation");
}

NeuralNetwork::NeuralNetwork(int hiddenSize, Activation activation, unsigned int seed)
    : inputSize_(1),
      hiddenSize_(hiddenSize),
      outputSize_(1),
      activation_(activation),
      rng_(seed) {
    W1_ = initializedMatrix(hiddenSize_, inputSize_, inputSize_);
    W2_ = initializedMatrix(hiddenSize_, hiddenSize_, hiddenSize_);
    W3_ = initializedMatrix(outputSize_, hiddenSize_, hiddenSize_);

    b1_ = Eigen::VectorXd::Zero(hiddenSize_);
    b2_ = Eigen::VectorXd::Zero(hiddenSize_);
    b3_ = Eigen::VectorXd::Zero(outputSize_);

    zeroGradients();
}

Eigen::MatrixXd NeuralNetwork::initializedMatrix(int rows, int cols, double fanIn) {
    // He initialization is a good default for ReLU. Tanh and sigmoid work
    // better with a smaller Xavier-style scale because large inputs push them
    // into saturated regions where their derivatives are close to zero.
    const double scale = activation_ == Activation::Relu ? 2.0 : 1.0;
    const double stddev = std::sqrt(scale / fanIn);
    std::normal_distribution<double> normal(0.0, stddev);

    Eigen::MatrixXd matrix(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            matrix(row, col) = normal(rng_);
        }
    }
    return matrix;
}

Eigen::VectorXd NeuralNetwork::activate(const Eigen::VectorXd& z) const {
    switch (activation_) {
        case Activation::Relu:
            return z.cwiseMax(0.0);
        case Activation::Tanh:
            return z.array().tanh().matrix();
        case Activation::Sigmoid:
            return (1.0 / (1.0 + (-z.array()).exp())).matrix();
    }

    throw std::invalid_argument("unknown activation");
}

Eigen::VectorXd NeuralNetwork::activationPrime(const Eigen::VectorXd& z) const {
    switch (activation_) {
        case Activation::Relu:
            return (z.array() > 0.0).cast<double>().matrix();
        case Activation::Tanh: {
            const Eigen::ArrayXd a = z.array().tanh();
            return (1.0 - a.square()).matrix();
        }
        case Activation::Sigmoid: {
            const Eigen::ArrayXd a = 1.0 / (1.0 + (-z.array()).exp());
            return (a * (1.0 - a)).matrix();
        }
    }

    throw std::invalid_argument("unknown activation");
}

Eigen::VectorXd NeuralNetwork::forward(const Eigen::VectorXd& x) {
    // Layer 0 is the scalar input stored as a length-1 vector.
    a0_ = x;

    // Hidden layer 1: affine transform followed by the selected activation.
    z1_ = W1_ * a0_ + b1_;
    a1_ = activate(z1_);

    // Hidden layer 2: affine transform followed by the selected activation.
    z2_ = W2_ * a1_ + b2_;
    a2_ = activate(z2_);

    // Output layer: affine transform only. This is regression, so the output
    // activation is linear and a3_ is the prediction.
    z3_ = W3_ * a2_ + b3_;
    a3_ = z3_;

    return a3_;
}

void NeuralNetwork::backward(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    const Eigen::VectorXd yHat = forward(x);

    // For mean squared error with a single output, the common factor of 2 can
    // be folded into the learning rate. The important chain-rule structure is
    // visible below, moving from output layer back to the first hidden layer.
    const Eigen::VectorXd delta3 = yHat - y;
    dW3_ += delta3 * a2_.transpose();
    db3_ += delta3;

    const Eigen::VectorXd delta2 =
        (W3_.transpose() * delta3).cwiseProduct(activationPrime(z2_));
    dW2_ += delta2 * a1_.transpose();
    db2_ += delta2;

    const Eigen::VectorXd delta1 =
        (W2_.transpose() * delta2).cwiseProduct(activationPrime(z1_));
    dW1_ += delta1 * a0_.transpose();
    db1_ += delta1;
}

void NeuralNetwork::zeroGradients() {
    dW1_ = Eigen::MatrixXd::Zero(hiddenSize_, inputSize_);
    dW2_ = Eigen::MatrixXd::Zero(hiddenSize_, hiddenSize_);
    dW3_ = Eigen::MatrixXd::Zero(outputSize_, hiddenSize_);

    db1_ = Eigen::VectorXd::Zero(hiddenSize_);
    db2_ = Eigen::VectorXd::Zero(hiddenSize_);
    db3_ = Eigen::VectorXd::Zero(outputSize_);
}

void NeuralNetwork::updateParameters(double learningRate, int batchSize) {
    const double scale = learningRate / static_cast<double>(batchSize);

    W1_ -= scale * dW1_;
    W2_ -= scale * dW2_;
    W3_ -= scale * dW3_;

    b1_ -= scale * db1_;
    b2_ -= scale * db2_;
    b3_ -= scale * db3_;
}

double NeuralNetwork::predict(double x) {
    Eigen::VectorXd input(1);
    input(0) = x;
    return forward(input)(0);
}

double NeuralNetwork::computeLoss(const Eigen::VectorXd& yHat, const Eigen::VectorXd& y) const {
    const Eigen::VectorXd diff = yHat - y;
    return diff.squaredNorm() / static_cast<double>(diff.size());
}

const Eigen::VectorXd& NeuralNetwork::output() const {
    return a3_;
}
