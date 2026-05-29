#include "neural_network.h"

#include <cmath>

NeuralNetwork::NeuralNetwork(int hiddenSize, unsigned int seed)
    : inputSize_(1),
      hiddenSize_(hiddenSize),
      outputSize_(1),
      rng_(seed) {
    W1_ = heMatrix(hiddenSize_, inputSize_, inputSize_);
    W2_ = heMatrix(hiddenSize_, hiddenSize_, hiddenSize_);
    W3_ = heMatrix(outputSize_, hiddenSize_, hiddenSize_);

    b1_ = Eigen::VectorXd::Zero(hiddenSize_);
    b2_ = Eigen::VectorXd::Zero(hiddenSize_);
    b3_ = Eigen::VectorXd::Zero(outputSize_);

    zeroGradients();
}

Eigen::MatrixXd NeuralNetwork::heMatrix(int rows, int cols, double fanIn) {
    const double stddev = std::sqrt(2.0 / fanIn);
    std::normal_distribution<double> normal(0.0, stddev);

    Eigen::MatrixXd matrix(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            matrix(row, col) = normal(rng_);
        }
    }
    return matrix;
}

Eigen::VectorXd NeuralNetwork::relu(const Eigen::VectorXd& z) const {
    return z.cwiseMax(0.0);
}

Eigen::VectorXd NeuralNetwork::reluPrime(const Eigen::VectorXd& z) const {
    return (z.array() > 0.0).cast<double>().matrix();
}

Eigen::VectorXd NeuralNetwork::forward(const Eigen::VectorXd& x) {
    // Layer 0 is the scalar input stored as a length-1 vector.
    a0_ = x;

    // Hidden layer 1: affine transform followed by ReLU.
    z1_ = W1_ * a0_ + b1_;
    a1_ = relu(z1_);

    // Hidden layer 2: affine transform followed by ReLU.
    z2_ = W2_ * a1_ + b2_;
    a2_ = relu(z2_);

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
        (W3_.transpose() * delta3).cwiseProduct(reluPrime(z2_));
    dW2_ += delta2 * a1_.transpose();
    db2_ += delta2;

    const Eigen::VectorXd delta1 =
        (W2_.transpose() * delta2).cwiseProduct(reluPrime(z1_));
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
