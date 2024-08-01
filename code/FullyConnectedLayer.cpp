
#include "FullyConnectedLayer.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm> // For std::max
#include <numeric>   // For std::accumulate

// Constructor - TEST OK
FullyConnectedLayer::FullyConnectedLayer(int numInputs, int numNeurons, ActivationFunction activation,
                                         LossFunction loss, bool useBias)
    : numInputs(numInputs), numNeurons(numNeurons), activationFunction(activation),
      lossFunction(loss), useBias(useBias) {
    // Allocate memory for weights and biases
    weights = new double[numInputs * numNeurons];
    biases = useBias ? new double[numNeurons] : nullptr;

    // Initialize weights and biases with random values
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int entryAndNeuronIndex = 0; entryAndNeuronIndex < numInputs * numNeurons; ++entryAndNeuronIndex) {
        weights[entryAndNeuronIndex] = distribution(generator);
    }

    if (useBias) {
        for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
            biases[neuronIndex] = distribution(generator);
        }
    }
}

// Destructor - TEST OK
FullyConnectedLayer::~FullyConnectedLayer() {
    delete[] weights;
    if (useBias) {
        delete[] biases;
    }
}

// Forward Pass
std::vector<double> FullyConnectedLayer::forward(const std::vector<double>& input) const{
    std::vector<double> output(numNeurons, 0.0);

    for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
        double neuronOutput = 0.0;
        for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex) {
          // the max Input EntryAmount In Order To Jump the next neuron is numInputs
            int entryAndNeuronIndex = (numInputs * neuronIndex) + entryIndex;
            neuronOutput += input[entryIndex] * weights[ entryAndNeuronIndex ];
        }
        if (useBias) {
            neuronOutput += biases[neuronIndex];
        }
        output[neuronIndex] = activate(neuronOutput);
    }

    // Apply softmax if chosen
    if (activationFunction == ActivationFunction::SOFTMAX) {
        softmax(output);
    }

    return output;
}


// Activation Function
double FullyConnectedLayer::activate(double zPreactivation) const {
    switch (activationFunction) {
        case ActivationFunction::RELU:
            return std::max(0.0, zPreactivation);
        case ActivationFunction::SIGMOID:
            return 1.0 / (1.0 + exp(-zPreactivation));
        case ActivationFunction::TANH:
            return std::tanh(zPreactivation);
        case ActivationFunction::LINEAR:
        case ActivationFunction::SOFTMAX:
            // Softmax will be handled separately for the entire vector
        default:
            return zPreactivation;
    }
}

// Softmax Function (vector-wise)
void FullyConnectedLayer::softmax(std::vector<double>& output) const {
    double maximunValue = *std::max_element(output.begin(), output.end());

    double sumExp = 0.0;
    for (auto& value : output) {
        value = exp(value - maximunValue);
        sumExp += value;
    }

    for (auto& value : output) {
        value /= sumExp;
    }
}


// Compute Loss
//expected == predicted
double FullyConnectedLayer::computeLoss(const std::vector<double>& input, const std::vector<double>& predicted) const {
    double loss = 0.0;
    std::vector<double> actual = this->forward(input);

    switch (lossFunction) {
        case LossFunction::MSE:
            for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
                double diff = actual[neuronIndex] - predicted[neuronIndex];
                loss += diff * diff;
            }
            loss /= numNeurons;
            break;

        case LossFunction::CROSS_ENTROPY: // Binary Cross Entropy
            for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
                loss -= predicted[neuronIndex] * std::log(actual[neuronIndex] + 1e-9) +
                        (1 - predicted[neuronIndex]) * std::log(1 - actual[neuronIndex] + 1e-9);
            }
            loss /= numNeurons;
            break;

        case LossFunction::CATEGORICAL_CROSS_ENTROPY:
            for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
                loss -= predicted[neuronIndex] * std::log(actual[neuronIndex] + 1e-9);
            }
            break;
    }
    return loss;
}

// Calculate Loss
double FullyConnectedLayer::calculateLoss(const std::vector<double>& input, const std::vector<double>& predicted) {
    if (input.size() != numInputs) {
        std::cerr << "Error: Size of input and numInputs must match.\n";
        return -1;
    }
    if (predicted.size() != numNeurons) {
        std::cerr << "Error: Size of predicted and numNeurons must match.\n";
        return -1;
    }
    return computeLoss(input, predicted);
}

// Getters (optional)
double* FullyConnectedLayer::getWeights() const {
    return weights;
}

double* FullyConnectedLayer::getBiases() const {
    return biases;
}

ActivationFunction FullyConnectedLayer::getActivationFunction() const {
    return activationFunction;
}

LossFunction FullyConnectedLayer::getLossFunction() const {
    return lossFunction;
}
