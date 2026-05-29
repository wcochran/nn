#include "mnist_loader.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {
constexpr int ImageMagic = 2051;
constexpr int LabelMagic = 2049;
constexpr int ImageRows = 28;
constexpr int ImageCols = 28;
constexpr int ImageSize = ImageRows * ImageCols;
constexpr int LabelCount = 10;

std::uint32_t readBigEndianUint32(std::ifstream& stream) {
    unsigned char bytes[4] = {0, 0, 0, 0};
    stream.read(reinterpret_cast<char*>(bytes), 4);
    if (!stream) {
        throw std::runtime_error("Unexpected end of IDX file while reading a 32-bit integer.");
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

Eigen::VectorXd oneHot(int label) {
    if (label < 0 || label >= LabelCount) {
        throw std::runtime_error("MNIST label is outside the expected range 0..9.");
    }

    Eigen::VectorXd target = Eigen::VectorXd::Zero(LabelCount);
    target(label) = 1.0;
    return target;
}
}

std::vector<MnistSample> loadMnistDataset(const std::string& imagePath, const std::string& labelPath) {
    std::ifstream images(imagePath, std::ios::binary);
    if (!images) {
        throw std::runtime_error("Could not open MNIST image file: " + imagePath);
    }

    std::ifstream labels(labelPath, std::ios::binary);
    if (!labels) {
        throw std::runtime_error("Could not open MNIST label file: " + labelPath);
    }

    const std::uint32_t imageMagic = readBigEndianUint32(images);
    const std::uint32_t imageCount = readBigEndianUint32(images);
    const std::uint32_t rows = readBigEndianUint32(images);
    const std::uint32_t cols = readBigEndianUint32(images);

    const std::uint32_t labelMagic = readBigEndianUint32(labels);
    const std::uint32_t labelCount = readBigEndianUint32(labels);

    if (imageMagic != ImageMagic) {
        throw std::runtime_error("Invalid MNIST image magic number in: " + imagePath);
    }
    if (labelMagic != LabelMagic) {
        throw std::runtime_error("Invalid MNIST label magic number in: " + labelPath);
    }
    if (rows != ImageRows || cols != ImageCols) {
        throw std::runtime_error("MNIST images are not 28 x 28 in: " + imagePath);
    }
    if (imageCount != labelCount) {
        throw std::runtime_error("MNIST image and label counts do not match.");
    }

    std::vector<MnistSample> samples;
    samples.reserve(imageCount);

    for (std::uint32_t sampleIndex = 0; sampleIndex < imageCount; ++sampleIndex) {
        Eigen::VectorXd input(ImageSize);
        for (int pixelIndex = 0; pixelIndex < ImageSize; ++pixelIndex) {
            unsigned char pixel = 0;
            images.read(reinterpret_cast<char*>(&pixel), 1);
            if (!images) {
                throw std::runtime_error("Unexpected end of MNIST image file: " + imagePath);
            }
            input(pixelIndex) = static_cast<double>(pixel) / 255.0;
        }

        unsigned char rawLabel = 0;
        labels.read(reinterpret_cast<char*>(&rawLabel), 1);
        if (!labels) {
            throw std::runtime_error("Unexpected end of MNIST label file: " + labelPath);
        }

        const int label = static_cast<int>(rawLabel);
        samples.push_back(MnistSample{input, oneHot(label), label});
    }

    return samples;
}
