#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

struct MnistSample {
    Eigen::VectorXd input;
    Eigen::VectorXd target;
    int label;
};

std::vector<MnistSample> loadMnistDataset(const std::string& imagePath, const std::string& labelPath);
