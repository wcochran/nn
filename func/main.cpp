#include "functions.h"
#include "neural_network.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

struct Config {
    std::string functionName = "erf";
    double xmin = -3.0;
    double xmax = 3.0;
    int hiddenSize = 16;
    int epochs = 5000;
    int batchSize = 32;
    double learningRate = 0.01;
    int samplesPerEpoch = 1024;
    int csvPoints = 401;
};

void printUsage(const char* programName) {
    std::cerr
        << "Usage: " << programName << " [function] [xmin] [xmax] [epochs] "
        << "[hidden_size] [batch_size] [learning_rate] [samples_per_epoch]\n"
        << "\nFunctions: ";

    const auto names = availableFunctionNames();
    for (std::size_t i = 0; i < names.size(); ++i) {
        std::cerr << names[i] << (i + 1 == names.size() ? "\n" : ", ");
    }

    std::cerr
        << "\nDefaults: erf -3 3 5000 16 32 0.01 1024\n"
        << "Output: predictions.csv\n";
}

Config parseArgs(int argc, char** argv) {
    Config config;

    if (argc > 1) {
        const std::string firstArg = argv[1];
        if (firstArg == "-h" || firstArg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        config.functionName = firstArg;
    }
    if (argc > 2) {
        config.xmin = std::stod(argv[2]);
    }
    if (argc > 3) {
        config.xmax = std::stod(argv[3]);
    }
    if (argc > 4) {
        config.epochs = std::stoi(argv[4]);
    }
    if (argc > 5) {
        config.hiddenSize = std::stoi(argv[5]);
    }
    if (argc > 6) {
        config.batchSize = std::stoi(argv[6]);
    }
    if (argc > 7) {
        config.learningRate = std::stod(argv[7]);
    }
    if (argc > 8) {
        config.samplesPerEpoch = std::stoi(argv[8]);
    }

    if (config.xmax <= config.xmin) {
        throw std::invalid_argument("xmax must be greater than xmin");
    }
    if (config.hiddenSize <= 0 || config.epochs <= 0 || config.batchSize <= 0 ||
        config.learningRate <= 0.0 || config.samplesPerEpoch <= 0) {
        throw std::invalid_argument("hidden size, epochs, batch size, learning rate, and samples per epoch must be positive");
    }

    return config;
}

void writePredictionsCsv(
    const std::string& path,
    NeuralNetwork& network,
    const TargetFunction& target,
    double xmin,
    double xmax,
    int points) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("could not open " + path + " for writing");
    }

    out << "x,true_y,predicted_y,error\n";
    out << std::setprecision(17);

    for (int i = 0; i < points; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(points - 1);
        const double x = xmin + t * (xmax - xmin);
        const double trueY = target.fn(x);
        const double predictedY = network.predict(x);
        out << x << ',' << trueY << ',' << predictedY << ',' << (predictedY - trueY) << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parseArgs(argc, argv);
        const TargetFunction target = getTargetFunction(config.functionName);

        NeuralNetwork network(config.hiddenSize);
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> xDistribution(config.xmin, config.xmax);

        const int batchesPerEpoch =
            std::max(1, config.samplesPerEpoch / config.batchSize);
        const int printEvery = std::max(1, config.epochs / 20);

        std::cout << "Training function '" << target.name << "' on ["
                  << config.xmin << ", " << config.xmax << "]\n"
                  << "hidden_size=" << config.hiddenSize
                  << ", epochs=" << config.epochs
                  << ", batch_size=" << config.batchSize
                  << ", learning_rate=" << config.learningRate
                  << ", samples_per_epoch=" << config.samplesPerEpoch << "\n";

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            double epochLoss = 0.0;
            int samplesSeen = 0;

            for (int batch = 0; batch < batchesPerEpoch; ++batch) {
                network.zeroGradients();

                for (int i = 0; i < config.batchSize; ++i) {
                    const double xValue = xDistribution(rng);
                    const double yValue = target.fn(xValue);

                    Eigen::VectorXd x(1);
                    Eigen::VectorXd y(1);
                    x(0) = xValue;
                    y(0) = yValue;

                    network.backward(x, y);
                    epochLoss += network.computeLoss(network.output(), y);
                    ++samplesSeen;
                }

                network.updateParameters(config.learningRate, config.batchSize);
            }

            if (epoch == 1 || epoch % printEvery == 0 || epoch == config.epochs) {
                std::cout << "epoch " << std::setw(5) << epoch
                          << " | average loss " << (epochLoss / samplesSeen) << '\n';
            }
        }

        writePredictionsCsv(
            "predictions.csv",
            network,
            target,
            config.xmin,
            config.xmax,
            config.csvPoints);

        std::cout << "Wrote predictions.csv\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}
