#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

struct TargetFunction {
    std::string name;
    std::function<double(double)> fn;
};

inline std::vector<std::string> availableFunctionNames() {
    return {"erf", "cos", "sin", "sin5", "gaussian", "square", "tanh"};
}

inline TargetFunction getTargetFunction(const std::string& name) {
    if (name == "erf") {
        return {name, [](double x) { return std::erf(x); }};
    }
    if (name == "cos") {
        return {name, [](double x) { return std::cos(x); }};
    }
    if (name == "sin") {
        return {name, [](double x) { return std::sin(x); }};
    }
    if (name == "sin5") {
        return {name, [](double x) { return std::sin(5.0 * x); }};
    }
    if (name == "gaussian") {
        return {name, [](double x) { return std::exp(-x * x); }};
    }
    if (name == "square") {
        return {name, [](double x) { return x * x; }};
    }
    if (name == "tanh") {
        return {name, [](double x) { return std::tanh(x); }};
    }

    throw std::invalid_argument("unknown function: " + name);
}

#endif
