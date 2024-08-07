/*
@Author: Francisco Adrian Llaryora

 The Class Dense represent a Fully connected ( a.k.a. FC) layer with n neurons.
 The class does not know if it is a hidden layer or an output layer.
 The model is neuron network on the hide layouts.
*/

#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <vector>

// Define enums for Activation and Loss Functions
enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    LINEAR,
    SOFTMAX
};

enum class LossFunction {
    MSE,  // Mean Squared Error
    CROSS_ENTROPY, // Binary Cross Entropy
    CATEGORICAL_CROSS_ENTROPY
};


class FullyConnectedLayer{

public:
  FullyConnectedLayer(
    int numInputs,
    int numNeurons,
    ActivationFunction activation,
    LossFunction loss,
    bool useBias
  );

  ~FullyConnectedLayer();

  // Public methods
  std::vector<double> forward(const std::vector<double>& input) const;
  double calculateLoss(
    const std::vector<double>& input,
    const std::vector<double>& predicted);

  // Getters for internal state (optional)
  double* getWeights() const;
  double* getBiases() const;
  ActivationFunction getActivationFunction() const;
  LossFunction getLossFunction() const;

 //private:
 protected:
     int numInputs;             // Number of input entries
     int numNeurons;            // Number of neurons in the layer
     bool useBias;              // Whether to use bias
     ActivationFunction activationFunction;
     LossFunction lossFunction;

     double* weights;           // Dynamically allocated array for weights
     double* biases;            // Dynamically allocated array for biases
     // Private methods
     double activate(double zPreactivation) const;
     std::vector<double> softmax(const std::vector<double>& zPreactivationArray) const;
     double computeLoss(const std::vector<double>& input, const std::vector<double>& predicted) const;
};

#endif //FULLYCONNECTEDLAYER_H
