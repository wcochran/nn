#include "neural_network.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

namespace {
constexpr int InputSize = 784;
constexpr int HiddenSize = 16;
constexpr int OutputSize = 10;

void fillNormal(Eigen::MatrixXd& matrix, double standardDeviation, std::mt19937& rng) {
    std::normal_distribution<double> distribution(0.0, standardDeviation);
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            matrix(row, col) = distribution(rng);
        }
    }
}
}

NeuralNetwork::NeuralNetwork()
    : W1(HiddenSize, InputSize),
      b1(Eigen::VectorXd::Zero(HiddenSize)),
      W2(HiddenSize, HiddenSize),
      b2(Eigen::VectorXd::Zero(HiddenSize)),
      W3(OutputSize, HiddenSize),
      b3(Eigen::VectorXd::Zero(OutputSize)),
      dW1(Eigen::MatrixXd::Zero(HiddenSize, InputSize)),
      db1(Eigen::VectorXd::Zero(HiddenSize)),
      dW2(Eigen::MatrixXd::Zero(HiddenSize, HiddenSize)),
      db2(Eigen::VectorXd::Zero(HiddenSize)),
      dW3(Eigen::MatrixXd::Zero(OutputSize, HiddenSize)),
      db3(Eigen::VectorXd::Zero(OutputSize)),
      a0(Eigen::VectorXd::Zero(InputSize)),
      z1(Eigen::VectorXd::Zero(HiddenSize)),
      a1(Eigen::VectorXd::Zero(HiddenSize)),
      z2(Eigen::VectorXd::Zero(HiddenSize)),
      a2(Eigen::VectorXd::Zero(HiddenSize)),
      z3(Eigen::VectorXd::Zero(OutputSize)),
      a3(Eigen::VectorXd::Zero(OutputSize)) {
    std::random_device rd;
    std::mt19937 rng(rd());

    fillNormal(W1, std::sqrt(2.0 / static_cast<double>(InputSize)), rng);
    fillNormal(W2, std::sqrt(2.0 / static_cast<double>(HiddenSize)), rng);
    fillNormal(W3, std::sqrt(2.0 / static_cast<double>(HiddenSize)), rng);
}

Eigen::VectorXd NeuralNetwork::forward(const Eigen::VectorXd& input) {
    assert(input.size() == InputSize);

    // a0 is the flattened 28 x 28 input image: shape 784.
    a0 = input;

    // z1 = W1 * a0 + b1: (16 x 784) * (784) + (16) = (16)
    z1 = W1 * a0 + b1;
    a1 = relu(z1);

    // z2 = W2 * a1 + b2: (16 x 16) * (16) + (16) = (16)
    z2 = W2 * a1 + b2;
    a2 = relu(z2);

    // z3 = W3 * a2 + b3: (10 x 16) * (16) + (10) = (10)
    z3 = W3 * a2 + b3;
    a3 = softmax(z3);

    return a3;
}

void NeuralNetwork::backward(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
    assert(input.size() == InputSize);
    assert(target.size() == OutputSize);

    forward(input);

    // For softmax followed by cross entropy, the derivative with respect to
    // z3 simplifies to output probabilities minus the one-hot target.
    // delta3 shape: 10.
    Eigen::VectorXd delta3 = a3 - target;

    // dW3 += delta3 * a2^T: (10) * (1 x 16) = (10 x 16)
    // db3 += delta3: (10)
    dW3 += delta3 * a2.transpose();
    db3 += delta3;

    // Chain rule through W3 and ReLU at z2:
    // W3^T * delta3: (16 x 10) * (10) = (16)
    // ReLUPrime(z2) gates gradients where hidden pre-activations were <= 0.
    Eigen::VectorXd delta2 = (W3.transpose() * delta3).cwiseProduct(reluPrime(z2));

    // dW2 += delta2 * a1^T: (16) * (1 x 16) = (16 x 16)
    // db2 += delta2: (16)
    dW2 += delta2 * a1.transpose();
    db2 += delta2;

    // Chain rule through W2 and ReLU at z1.
    // W2^T * delta2: (16 x 16) * (16) = (16)
    Eigen::VectorXd delta1 = (W2.transpose() * delta2).cwiseProduct(reluPrime(z1));

    // dW1 += delta1 * a0^T: (16) * (1 x 784) = (16 x 784)
    // db1 += delta1: (16)
    dW1 += delta1 * a0.transpose();
    db1 += delta1;
}

void NeuralNetwork::zeroGradients() {
    dW1.setZero();
    db1.setZero();
    dW2.setZero();
    db2.setZero();
    dW3.setZero();
    db3.setZero();
}

void NeuralNetwork::updateParameters(double learningRate, int batchSize) {
    assert(batchSize > 0);
    const double scale = learningRate / static_cast<double>(batchSize);

    W1 -= scale * dW1;
    b1 -= scale * db1;
    W2 -= scale * dW2;
    b2 -= scale * db2;
    W3 -= scale * dW3;
    b3 -= scale * db3;
}

int NeuralNetwork::predict(const Eigen::VectorXd& input) {
    Eigen::VectorXd output = forward(input);
    Eigen::Index predictedIndex = 0;
    output.maxCoeff(&predictedIndex);
    return static_cast<int>(predictedIndex);
}

double NeuralNetwork::computeLoss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const {
    assert(output.size() == OutputSize);
    assert(target.size() == OutputSize);

    constexpr double Epsilon = 1e-12;
    double loss = 0.0;
    for (int i = 0; i < OutputSize; ++i) {
        loss -= target(i) * std::log(std::max(output(i), Epsilon));
    }
    return loss;
}

Eigen::VectorXd NeuralNetwork::relu(const Eigen::VectorXd& z) const {
    return z.cwiseMax(0.0);
}

Eigen::VectorXd NeuralNetwork::reluPrime(const Eigen::VectorXd& z) const {
    Eigen::VectorXd derivative(z.size());
    for (int i = 0; i < z.size(); ++i) {
        derivative(i) = z(i) > 0.0 ? 1.0 : 0.0;
    }
    return derivative;
}

Eigen::VectorXd NeuralNetwork::softmax(const Eigen::VectorXd& z) const {
    assert(z.size() == OutputSize);

    // Subtracting max(z) leaves the probabilities unchanged but prevents
    // overflow when exponentiating large logits.
    const double maxLogit = z.maxCoeff();
    Eigen::VectorXd shifted = z.array() - maxLogit;
    Eigen::VectorXd expValues = shifted.array().exp();
    return expValues / expValues.sum();
}
