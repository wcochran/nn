#include "mnist_loader.h"
#include "neural_network.h"

#include <Eigen/Dense>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int InputSize = 784;
constexpr int OutputSize = 10;

std::vector<MnistSample> makeSyntheticSamples() {
    std::vector<MnistSample> samples;
    samples.reserve(10);

    for (int label = 0; label < OutputSize; ++label) {
        Eigen::VectorXd input = Eigen::VectorXd::Zero(InputSize);

        // A tiny deterministic toy pattern: each class lights a different
        // small stripe. This only demonstrates that training code runs.
        const int start = label * 70;
        for (int i = start; i < std::min(start + 70, InputSize); ++i) {
            input(i) = 1.0;
        }

        Eigen::VectorXd target = Eigen::VectorXd::Zero(OutputSize);
        target(label) = 1.0;
        samples.push_back(MnistSample{input, target, label});
    }

    return samples;
}

bool tryLoadMnist(std::vector<MnistSample>& trainSamples, std::vector<MnistSample>& testSamples) {
    try {
        trainSamples = loadMnistDataset("data/MNIST/raw/train-images-idx3-ubyte", "data/MNIST/raw/train-labels-idx1-ubyte");
        testSamples = loadMnistDataset("data/MNIST/raw/t10k-images-idx3-ubyte", "data/MNIST/raw/t10k-labels-idx1-ubyte");
        return true;
    } catch (const std::exception& error) {
        std::cout << "MNIST files not loaded: " << error.what() << '\n';
        std::cout << "Run ./fetch_mnist.sh to download the shared IDX files into data/MNIST/raw." << '\n';
        std::cout << "Using a tiny synthetic dataset instead.\n";
        return false;
    }
}

double accuracy(NeuralNetwork& network, const std::vector<MnistSample>& samples) {
    int correct = 0;
    for (const MnistSample& sample : samples) {
        if (network.predict(sample.input) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(samples.size());
}

double averageLoss(NeuralNetwork& network, const std::vector<MnistSample>& samples) {
    double totalLoss = 0.0;
    for (const MnistSample& sample : samples) {
        Eigen::VectorXd output = network.forward(sample.input);
        totalLoss += network.computeLoss(output, sample.target);
    }
    return totalLoss / static_cast<double>(samples.size());
}
}

int main() {
    std::vector<MnistSample> trainSamples;
    std::vector<MnistSample> testSamples;
    const bool usingMnist = tryLoadMnist(trainSamples, testSamples);

    if (!usingMnist) {
        trainSamples = makeSyntheticSamples();
        testSamples = trainSamples;
    }

    NeuralNetwork network;

    const int epochs = usingMnist ? 5 : 30;
    const int batchSize = usingMnist ? 32 : 5;
    const double learningRate = usingMnist ? 0.05 : 0.1;

    std::vector<int> order(trainSamples.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(std::random_device{}());

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);

        for (std::size_t batchStart = 0; batchStart < order.size(); batchStart += batchSize) {
            network.zeroGradients();

            const std::size_t batchEnd = std::min(batchStart + static_cast<std::size_t>(batchSize), order.size());
            for (std::size_t index = batchStart; index < batchEnd; ++index) {
                const MnistSample& sample = trainSamples[order[index]];
                network.backward(sample.input, sample.target);
            }

            network.updateParameters(learningRate, static_cast<int>(batchEnd - batchStart));
        }

        const double trainLoss = averageLoss(network, trainSamples);
        const double trainAccuracy = accuracy(network, trainSamples);
        const double testAccuracy = accuracy(network, testSamples);

        std::cout << "Epoch " << epoch
                  << " | loss: " << trainLoss
                  << " | train accuracy: " << trainAccuracy * 100.0 << "%"
                  << " | test accuracy: " << testAccuracy * 100.0 << "%\n";
    }

    return 0;
}
