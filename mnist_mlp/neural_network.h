#pragma once

#include <Eigen/Dense>

class NeuralNetwork {
public:
    NeuralNetwork();

    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    void backward(const Eigen::VectorXd& input, const Eigen::VectorXd& target);
    void zeroGradients();
    void updateParameters(double learningRate, int batchSize);
    int predict(const Eigen::VectorXd& input);
    double computeLoss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const;

private:
    Eigen::VectorXd relu(const Eigen::VectorXd& z) const;
    Eigen::VectorXd reluPrime(const Eigen::VectorXd& z) const;
    Eigen::VectorXd softmax(const Eigen::VectorXd& z) const;

    // W1: 16 x 784, b1: 16
    // W2: 16 x 16,  b2: 16
    // W3: 10 x 16,  b3: 10
    Eigen::MatrixXd W1;
    Eigen::VectorXd b1;
    Eigen::MatrixXd W2;
    Eigen::VectorXd b2;
    Eigen::MatrixXd W3;
    Eigen::VectorXd b3;

    Eigen::MatrixXd dW1;
    Eigen::VectorXd db1;
    Eigen::MatrixXd dW2;
    Eigen::VectorXd db2;
    Eigen::MatrixXd dW3;
    Eigen::VectorXd db3;

    // Cached values from the most recent forward pass.
    Eigen::VectorXd a0;
    Eigen::VectorXd z1;
    Eigen::VectorXd a1;
    Eigen::VectorXd z2;
    Eigen::VectorXd a2;
    Eigen::VectorXd z3;
    Eigen::VectorXd a3;
};
