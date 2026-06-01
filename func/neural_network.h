#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Eigen/Dense>
#include <random>
#include <string>
#include <vector>


enum class Activation {
    Relu,
    Tanh,
    Sigmoid
};

std::vector<std::string> availableActivationNames();
Activation activationFromName(const std::string& name);
std::string activationName(Activation activation);

class NeuralNetwork {
public:
    explicit NeuralNetwork(
        int hiddenSize = 16,
        Activation activation = Activation::Relu,
        unsigned int seed = 42);

    Eigen::VectorXd forward(const Eigen::VectorXd& x);
    void backward(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    void zeroGradients();
    void updateParameters(double learningRate, int batchSize);
    double predict(double x);
    double computeLoss(const Eigen::VectorXd& yHat, const Eigen::VectorXd& y) const;
    const Eigen::VectorXd& output() const;

private:
    int inputSize_;
    int hiddenSize_;
    int outputSize_;
    Activation activation_;

    Eigen::MatrixXd W1_;
    Eigen::MatrixXd W2_;
    Eigen::MatrixXd W3_;
    Eigen::VectorXd b1_;
    Eigen::VectorXd b2_;
    Eigen::VectorXd b3_;

    Eigen::MatrixXd dW1_;
    Eigen::MatrixXd dW2_;
    Eigen::MatrixXd dW3_;
    Eigen::VectorXd db1_;
    Eigen::VectorXd db2_;
    Eigen::VectorXd db3_;

    // Cached values from the most recent forward pass. Keeping these visible
    // in the class makes the chain rule in backward() easy to follow.
    Eigen::VectorXd a0_;
    Eigen::VectorXd z1_;
    Eigen::VectorXd a1_;
    Eigen::VectorXd z2_;
    Eigen::VectorXd a2_;
    Eigen::VectorXd z3_;
    Eigen::VectorXd a3_;

    std::mt19937 rng_;

    Eigen::MatrixXd initializedMatrix(int rows, int cols, double fanIn);
    Eigen::VectorXd activate(const Eigen::VectorXd& z) const;
    Eigen::VectorXd activationPrime(const Eigen::VectorXd& z) const;
};

#endif
