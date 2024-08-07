#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "FullyConnectedLayer.h"

//  g++ -o fully_connected_layer test-dense.cpp FullyConnectedLayer.cpp
 //  ./fully_connected_layer

// Function to compare two vectors with a tolerance
bool compareVectors(const std::vector<double>& a, const std::vector<double>& b, double tolerance = 1e-6) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) return false;
    }
    return true;
}

// Debugging helper function to print vectors
void printVector(const std::vector<double>& vec, const std::string& label) {
    std::cout << label << ": [ ";
    for (const auto& v : vec) {
        std::cout << v << " ";
    }
    std::cout << "]" << std::endl;
}

int getEntryAndNeuronIndex(int entryIndex, int neuronIndex, int numInputs){
    return (numInputs * neuronIndex) + entryIndex;
}

// Test Initialization - PASS
void testInitialization() {
    std::cout << "Running testInitialization..." << std::endl;
    int numInputs = 3;
    int numNeurons = 2;
    FullyConnectedLayer layer(numInputs, numNeurons, ActivationFunction::RELU, LossFunction::MSE, true);

    // Check if weights and biases are allocated
    assert(layer.getWeights() != nullptr);
    assert(layer.getBiases() != nullptr);

    double* layerWeights = layer.getWeights();
    std::cout << "Vector layerWeights : [ ";
    for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex ) {
      for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex ) {
        int entryAndNeuronIndex = (numInputs * neuronIndex) + entryIndex;
        std::cout << layerWeights[entryAndNeuronIndex] << " ";
      }
    }
    std::cout << "]" << std::endl;

    double* layerBiases = layer.getBiases();
    std::cout << "Vector layerBiases : [ ";
    for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex ) {
        std::cout << layerBiases[neuronIndex] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "testInitialization passed." << std::endl;
}

// Test  - PASS
void tLineal(){
  int numInputs = 3;
  int numNeurons = 2;
  // Test Linear
  FullyConnectedLayer linearLayer(numInputs, numNeurons, ActivationFunction::LINEAR, LossFunction::MSE, false);
  double* linearWeights = linearLayer.getWeights();

  assert(linearLayer.getBiases() == nullptr);
  linearWeights[getEntryAndNeuronIndex(0, 0, numInputs)] = 0.5;
  linearWeights[getEntryAndNeuronIndex(1, 0, numInputs)] = 1.0;
  linearWeights[getEntryAndNeuronIndex(2, 0, numInputs)] = -0.5;
  linearWeights[getEntryAndNeuronIndex(0, 1, numInputs)] = -0.5;
  linearWeights[getEntryAndNeuronIndex(1, 1, numInputs)] = -1.0;
  linearWeights[getEntryAndNeuronIndex(2, 1, numInputs)] = 1.0;

  std::vector<double> linearInput = {2.0, -1.0, 3.0};
  std::vector<double> expectedLinearOutput = {-1.5, 3};
  std::vector<double> linearOutput = linearLayer.forward(linearInput);

  // Debugging output for additional test
  printVector(linearInput, "LINEAR Input (Positive Test)");
  printVector(linearOutput, "LINEAR Output (Positive Test)");
  printVector(expectedLinearOutput, "Expected LINEAR Output (Positive Test)");

  assert(compareVectors(linearOutput, expectedLinearOutput));

}

// Test  - PASS
void tLinealBias(){
  int numInputs = 3;
  int numNeurons = 2;
  // Test Linear
  FullyConnectedLayer linearLayer(numInputs, numNeurons, ActivationFunction::LINEAR, LossFunction::MSE, true);
  double* linearWeights = linearLayer.getWeights();

  linearWeights[getEntryAndNeuronIndex(0, 0, numInputs)] = 0.5;
  linearWeights[getEntryAndNeuronIndex(1, 0, numInputs)] = 1.0;
  linearWeights[getEntryAndNeuronIndex(2, 0, numInputs)] = -0.5;
  linearWeights[getEntryAndNeuronIndex(0, 1, numInputs)] = -0.5;
  linearWeights[getEntryAndNeuronIndex(1, 1, numInputs)] = -1.0;
  linearWeights[getEntryAndNeuronIndex(2, 1, numInputs)] = 1.0;

  double* layerBiases = linearLayer.getBiases();
  layerBiases[0] = 0.5;
  layerBiases[1] = -0.5;

  std::vector<double> linearInput = {2.0, -1.0, 3.0};
  std::vector<double> expectedLinearOutput = {-1.0, 2.5};
  std::vector<double> linearOutput = linearLayer.forward(linearInput);

  // Debugging output for additional test
  printVector(linearInput, "BIAS LINEAR Input (Positive Test)");
  printVector(linearOutput, "BIAS LINEAR Output (Positive Test)");
  printVector(expectedLinearOutput, "Expected BIAS LINEAR Output (Positive Test)");

  assert(compareVectors(linearOutput, expectedLinearOutput));

}

// Test  - PASS
void tSigmoidal(){
  // Test Sigmoid
  FullyConnectedLayer sigmoidLayer(1, 1, ActivationFunction::SIGMOID, LossFunction::MSE, false);
  double* sigmoidWeights = sigmoidLayer.getWeights();
  sigmoidWeights[0] = 0.0; // Input weight to the neuron
  std::vector<double> sigmoidInput = {0.0};
  std::vector<double> expectedSigmoidOutput = {0.5}; // Sigmoid(0) = 0.5
  std::vector<double> sigmoidOutput = sigmoidLayer.forward(sigmoidInput);
  assert(compareVectors(sigmoidOutput, expectedSigmoidOutput));
  std::cout << "SIGMOID test passed successfully!" << std::endl;
}

// Test  - PASS
void tTanH(){
  // Test Tanh
  FullyConnectedLayer tanhLayer(1, 1, ActivationFunction::TANH, LossFunction::MSE, false);
  double* tanhWeights = tanhLayer.getWeights();
  tanhWeights[0] = 0.0; // Input weight to the neuron
  std::vector<double> tanhInput = {0.0};
  std::vector<double> expectedTanhOutput = {0.0}; // Tanh(0) = 0.0
  std::vector<double> tanhOutput = tanhLayer.forward(tanhInput);
  assert(compareVectors(tanhOutput, expectedTanhOutput));
  std::cout << "TANH test passed successfully!" << std::endl;
}

// Test  - PASS
void tSoftmax(){
  // Test Softmax
  FullyConnectedLayer softmaxLayer(3, 3, ActivationFunction::SOFTMAX, LossFunction::CATEGORICAL_CROSS_ENTROPY, false);
  double* softmaxWeights = softmaxLayer.getWeights();
  softmaxWeights[0] = 1.0; // For input 0 -> neuron 0
  softmaxWeights[1] = 0.0; // For input 1 -> neuron 0
  softmaxWeights[2] = 0.0; // For input 2 -> neuron 0
  softmaxWeights[3] = 0.0; // For input 0 -> neuron 1
  softmaxWeights[4] = 1.0; // For input 1 -> neuron 1
  softmaxWeights[5] = 0.0; // For input 2 -> neuron 1
  softmaxWeights[6] = 0.0; // For input 0 -> neuron 2
  softmaxWeights[7] = 0.0; // For input 1 -> neuron 2
  softmaxWeights[8] = 1.0; // For input 2 -> neuron 2
  std::vector<double> softmaxInput = {1.0, 2.0, 3.0};
  std::vector<double> expectedSoftmaxOutput = {0.0900306, 0.244728, 0.665241}; // Known softmax outputs for these inputs
  std::vector<double> softmaxOutput = softmaxLayer.forward(softmaxInput);
  printVector(softmaxOutput, "Softmax Output");
  assert(compareVectors(softmaxOutput, expectedSoftmaxOutput));
  std::cout << "SOFTMAX test passed successfully!" << std::endl;
}

// Test  - PASS
void tReLU() {
  int numInputs = 3;
  int numNeurons = 3;
  // Test ReLU
  FullyConnectedLayer reluLayer(numInputs, numNeurons, ActivationFunction::RELU, LossFunction::MSE, false);

  // Manually set weights
  double* reluWeights = reluLayer.getWeights();
  reluWeights[0] = -1.0; // Weight for input 0 -> neuron 0
  reluWeights[1] = 0.0;  // Weight for input 1 -> neuron 0
  reluWeights[2] = 1.0;  // Weight for input 2 -> neuron 0
  reluWeights[3] = -1.0; // Weight for input 0 -> neuron 1
  reluWeights[4] = 0.0;  // Weight for input 1 -> neuron 1
  reluWeights[5] = 1.0;  // Weight for input 2 -> neuron 1
  reluWeights[6] = -1.0; // Weight for input 0 -> neuron 2
  reluWeights[7] = 0.0;  // Weight for input 1 -> neuron 2
  reluWeights[8] = 1.0;  // Weight for input 2 -> neuron 2

  std::vector<double> reluInput = {1.0, 1.0, 1.0};  // Input vector

  std::vector<double> expectedReluOutput = {0.0, 0.0, 0.0}; // Expected ReLU outputs

  std::vector<double> reluOutput = reluLayer.forward(reluInput);

  assert(compareVectors(reluOutput, expectedReluOutput));

  // Additional Test with Non-negative Outputs
  std::vector<double> reluInputPositive = {2.0, 3.0, -4.0};  // Input with mixed values
  std::vector<double> expectedReluOutputPositive = {0.0, 3.0, 0.0}; // Expected ReLU output for positive inputs

  reluWeights[getEntryAndNeuronIndex(0, 0, numInputs)] = -1.0;
  reluWeights[getEntryAndNeuronIndex(1, 0, numInputs)] = -1.0;
  reluWeights[getEntryAndNeuronIndex(2, 0, numInputs)] = 0.0;

  reluWeights[getEntryAndNeuronIndex(0, 1, numInputs)] = 0.0;
  reluWeights[getEntryAndNeuronIndex(1, 1, numInputs)] = 1.0;
  reluWeights[getEntryAndNeuronIndex(2, 1, numInputs)] = 0.0;

  reluWeights[getEntryAndNeuronIndex(0, 2, numInputs)] = 0.0;
  reluWeights[getEntryAndNeuronIndex(1, 2, numInputs)] = 0.0;
  reluWeights[getEntryAndNeuronIndex(2, 2, numInputs)] = 0.0;

  reluOutput = reluLayer.forward(reluInputPositive);

  // Debugging output for additional test
  printVector(reluInputPositive, "ReLU Input (Positive Test)");
  printVector(reluOutput, "ReLU Output (Positive Test)");
  printVector(expectedReluOutputPositive, "Expected ReLU Output (Positive Test)");

  assert(compareVectors(reluOutput, expectedReluOutputPositive));

}

// Test Activation Functions- Pass
void testActivationFunctions() {
    std::cout << "Running testActivationFunctions..." << std::endl;
    tLineal();
    tLinealBias();
    tSigmoidal();
    tTanH();
    tSoftmax();
    tReLU();

    std::cout << "testActivationFunctions passed." << std::endl;
}
// TEST PASS
void tMSE() {
  // Test MSE Loss
  FullyConnectedLayer mseLayer(3, 2, ActivationFunction::LINEAR, LossFunction::MSE, false);
  double* linearWeights = mseLayer.getWeights();
  int numInputs = 3;
  linearWeights[getEntryAndNeuronIndex(0, 0, numInputs)] = 0.5;
  linearWeights[getEntryAndNeuronIndex(1, 0, numInputs)] = 1.0;
  linearWeights[getEntryAndNeuronIndex(2, 0, numInputs)] = -0.5;
  linearWeights[getEntryAndNeuronIndex(0, 1, numInputs)] = -0.5;
  linearWeights[getEntryAndNeuronIndex(1, 1, numInputs)] = -1.0;
  linearWeights[getEntryAndNeuronIndex(2, 1, numInputs)] = 1.0;

  std::vector<double> linearInput = {2.0, -1.0, 3.0};
  std::vector<double> predictedLinearOutput = {-1.5, 3};

  std::vector<double> expectedMSE = {0.5, 0.5};

  double expectedMSELoss = 5.125;
  double mseLoss = mseLayer.calculateLoss(linearInput, expectedMSE);
  assert(std::abs(mseLoss - expectedMSELoss) < 1e-6);

  std::cout << "MSE passed." << std::endl;

}

void tBinaryCROSS(){
  // Test Binary Cross Entropy
  FullyConnectedLayer binaryLayer(1, 1, ActivationFunction::SIGMOID, LossFunction::CROSS_ENTROPY, false);
  double* sigmoidWeights = binaryLayer.getWeights();
  sigmoidWeights[0] = 0.0; // Input weight to the neuron
  std::vector<double> sigmoidInput = {0.0};
  std::vector<double> predictedSigmoidOutput = {0.5}; // Sigmoid(0) = 0.5
  std::vector<double> expectedBinary = {1.0};

  double expectedBinaryLoss = expectedBinary[0] * std::log(predictedSigmoidOutput[0] + 1e-9) +
          (1 - expectedBinary[0]) * std::log(1 - predictedSigmoidOutput[0] + 1e-9);
  expectedBinaryLoss *= -1.0;

  std::cout << "expectedBinaryLoss " << expectedBinaryLoss <<  std::endl;
  double binaryLoss = binaryLayer.calculateLoss(sigmoidInput, expectedBinary);
  std::cout << "binaryLoss " << binaryLoss <<  std::endl;
  assert(std::abs(binaryLoss - expectedBinaryLoss) < 1e-6);

  std::cout << "CROSS_ENTROPY passed." << std::endl;
}

// Test Loss Functions
void testLossFunctions() {
    std::cout << "Running testLossFunctions..." << std::endl;
    tMSE();
    tBinaryCROSS();

    // Test Categorical Cross Entropy
    FullyConnectedLayer categoricalLayer(3, 3, ActivationFunction::SOFTMAX, LossFunction::CATEGORICAL_CROSS_ENTROPY, false);
    double* softmaxWeights = categoricalLayer.getWeights();
    softmaxWeights[0] = 1.0; // For input 0 -> neuron 0
    softmaxWeights[1] = 0.0; // For input 1 -> neuron 0
    softmaxWeights[2] = 0.0; // For input 2 -> neuron 0
    softmaxWeights[3] = 0.0; // For input 0 -> neuron 1
    softmaxWeights[4] = 1.0; // For input 1 -> neuron 1
    softmaxWeights[5] = 0.0; // For input 2 -> neuron 1
    softmaxWeights[6] = 0.0; // For input 0 -> neuron 2
    softmaxWeights[7] = 0.0; // For input 1 -> neuron 2
    softmaxWeights[8] = 1.0; // For input 2 -> neuron 2
    std::vector<double> softmaxInput = {1.0, 2.0, 3.0};
    std::vector<double> predictedSoftmaxOutput = {0.0900306, 0.244728, 0.665241}; // Known softmax outputs for these inputs
    std::vector<double> expectedCategorical = {0.0, 0.0, 1.0};

    double expectedCategoricalLoss = -std::log(0.665241); // -[0*log(0.09) + 0*log(0.24) + 1*log(0.66)]
    double categoricalLoss = categoricalLayer.calculateLoss(softmaxInput, expectedCategorical);
    assert(std::abs(categoricalLoss - expectedCategoricalLoss) < 1e-6);

    std::cout << "testLossFunctions passed." << std::endl;
}

// Main function to run all tests
int main() {
    testInitialization();
    testActivationFunctions();
    testLossFunctions();

    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
