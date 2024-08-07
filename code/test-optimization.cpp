#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "FullyConnectedLayer.h"
#include "OptimizedFullyConnectedLayer.h"

//  g++ -o fully_fully  FullyConnectedLayer.cpp OptimizedFullyConnectedLayer.cpp test-optimization.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>


// Helper function to print vectors (for debugging)
void printVector(const std::vector<double>& vec) {
    for (double val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// Helper function to compare two vectors
bool vectorsAreEqual(const std::vector<double>& vec1, const std::vector<double>& vec2, double tolerance = 1e-6) {
    if (vec1.size() != vec2.size()) return false;
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::fabs(vec1[i] - vec2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int getEntryAndNeuronIndex(int entryIndex, int neuronIndex, int numInputs){
    return (numInputs * neuronIndex) + entryIndex;
}

//TEST OK
void testPreactivation() {
  // Define parameters
  int numInputs = 3;
  int numNeurons = 2;
  ActivationFunction activation = ActivationFunction::SIGMOID;
  LossFunction loss = LossFunction::MSE;
  bool useBias = true;
  OptimizationAlgorithm optimization = OptimizationAlgorithm::SGD;
  // Initialize the layer
  OptimizedFullyConnectedLayer layer(numInputs, numNeurons, activation, loss, useBias, optimization);

  // Manually set weights and biases for deterministic behavior
  double* weights = layer.getWeights();
  double* biases = layer.getBiases();

  // Example weights
  weights[getEntryAndNeuronIndex(0,0,numInputs)] = 0.1;
  weights[getEntryAndNeuronIndex(1,0,numInputs)] = 0.2;
  weights[getEntryAndNeuronIndex(2,0,numInputs)] = 0.3;
  weights[getEntryAndNeuronIndex(0,1,numInputs)] = 0.4;
  weights[getEntryAndNeuronIndex(1,1,numInputs)] = 0.5;
  weights[getEntryAndNeuronIndex(2,1,numInputs)] = 0.6;

  // Example biases
  biases[0] = 0.1;
  biases[1] = 0.2;

  // Define input and expected output
  std::vector<double> input = {1.0, 2.0, 3.0};

  std::vector<double> actualZ = layer.zPreactivation(input);
  std::vector<double> expectedZ = {1.5, 3.4};

  std::cout << "zPreactivation for input: ";
  printVector(actualZ);
  assert(vectorsAreEqual(actualZ, expectedZ, 1e-9));
}

//TEST OK
void testActivationDerivative() {
  int numInputs = 3;
  int numNeurons = 1;  // Single neuron for simplicity
  OptimizationAlgorithm optimization = OptimizationAlgorithm::SGD;
  // Create test input and expected output for each activation
  struct TestCase {
      ActivationFunction activation;
      std::vector<double> zPreactivationArray;
      double expectedDerivative;
  };

  std::vector<TestCase> testCases = {
        // RELU: derivative is 1.0 if z > 0 else 0.0
        {ActivationFunction::RELU, {1.0}, 1.0},   // Positive input
        {ActivationFunction::RELU, {-1.0}, 0.0},  // Negative input

        // SIGMOID: derivative is z * (1 - z)
        {ActivationFunction::SIGMOID, {0.5}, 0.5 * (1.0 - 0.5)},  // Sigmoid output

        // TANH: derivative is 1 - tanh(z)^2
        {ActivationFunction::TANH, {0.5}, 1.0 - std::tanh(0.5) * std::tanh(0.5)},

        // SOFTMAX: derivative is softmax(z) * (1 - softmax(z))
        // Use a vector since softmax works on the entire array
        {ActivationFunction::SOFTMAX, {1.0}, 0.0},

        // LINEAR: derivative is 1.0
        {ActivationFunction::LINEAR, {2.0}, 1.0}
    };

    for (const auto& testCase : testCases) {
        // Create a FullyConnectedLayer instance
        OptimizedFullyConnectedLayer layer(
            numInputs,
            numNeurons,
            testCase.activation,
            LossFunction::MSE, // Loss type doesn't matter for this test
            false,              // Bias doesn't affect activation derivatives
            optimization
        );

        // Compute the derivative
        double derivative = layer.activationDerivative(testCase.zPreactivationArray, 0);

        // Print the result for debugging
        std::cout << "Activation: ";
        switch (testCase.activation) {
            case ActivationFunction::RELU: std::cout << "RELU"; break;
            case ActivationFunction::SIGMOID: std::cout << "SIGMOID"; break;
            case ActivationFunction::TANH: std::cout << "TANH"; break;
            case ActivationFunction::SOFTMAX: std::cout << "SOFTMAX"; break;
            case ActivationFunction::LINEAR: std::cout << "LINEAR"; break;
        }
        std::cout << ", Input: " << testCase.zPreactivationArray[0]
                  << ", Expected Derivative: " << testCase.expectedDerivative
                  << ", Computed Derivative: " << derivative << std::endl;

        // Check if the computed derivative matches the expected value
        assert(std::fabs(derivative - testCase.expectedDerivative) < 1e-6);
    }

    std::cout << "All activation derivative tests passed!" << std::endl;
}

//TEST OK
void testLossGradients() {
  int numInputs = 3;
  int numNeurons = 1;  // Single neuron for simplicity
  ActivationFunction activation = ActivationFunction::SIGMOID; //necessary
  OptimizationAlgorithm optimization = OptimizationAlgorithm::SGD;
  // Create test input and expected output for each activation
  struct TestCase {
      LossFunction lossFunction;
      std::vector<double> actual;
      std::vector<double> expected;
      std::vector<double> expectedOutput;
  };

  std::vector<TestCase> testCases = {
        {LossFunction::MSE,{1.0},{0.7},{0.3}},// actual-expected

        {LossFunction::CATEGORICAL_CROSS_ENTROPY, {1.0},{0.7},{-0.7}},
        // -expected[neuronIndex] / (actual[neuronIndex] + 1e-9);
        // -0.7 / (1.0 + 1e-9);

        //-(actual / (expected + 1e-9)- (1 - actual / (1 - expected + 1e-9));
        //-( 1.0 / (0.7 + 1e-9) - (1 - 1,0) / (1 - 0.7 + 1e-9));
        //-( 1.428571427 - (0) / (0.3));
        ////-( 1.428571427 - 0);
        {LossFunction::CROSS_ENTROPY,{1.0},{0.7},{-1.428571427}}
    };

    for (const auto& testCase : testCases) {
        // Create a FullyConnectedLayer instance
        OptimizedFullyConnectedLayer layer(
            numInputs,
            numNeurons,
            activation,
            testCase.lossFunction,
            false,
            optimization
        );

        // Compute the derivative
        std::vector<double> derivative = layer.lossDerivative(testCase.actual, testCase.expected);

        // Print the result for debugging
        std::cout << "Loss: ";
        switch (testCase.lossFunction) {
            case LossFunction::MSE: std::cout << "MSE"; break;
            case LossFunction::CROSS_ENTROPY: std::cout << "BINARY_CROSS_ENTROPY"; break;
            case LossFunction::CATEGORICAL_CROSS_ENTROPY: std::cout << "CATEGORICAL_CROSS_ENTROPY"; break;
        }
        std::cout << ", Input: ";
        printVector(testCase.actual);
        printVector(testCase.expected);

        std::cout << ", Expected Derivative: ";
        printVector(testCase.expectedOutput);
        std::cout << ", Computed Derivative: ";
        printVector(derivative);

        // Check if the computed derivative matches the expected value
        assert(vectorsAreEqual(derivative, testCase.expectedOutput, 1e-9));
    }

    std::cout << "All loss derivative tests passed!" << std::endl;
}

//TEST OK
void testGenerateDeltas() {
  int numInputs = 3;
  int numNeurons = 1;  // Single neuron for simplicity
  ActivationFunction activation = ActivationFunction::SIGMOID; //necessary
  OptimizationAlgorithm optimization = OptimizationAlgorithm::SGD;
  LossFunction loss = LossFunction::CROSS_ENTROPY;

  // Create a FullyConnectedLayer instance
  OptimizedFullyConnectedLayer layer(
      numInputs,
      numNeurons,
      activation,
      loss,
      false,
      optimization
  );
  //{ActivationFunction::SIGMOID,
  // z = {0.5},
  // expected =  0.5 * (1.0 - 0.5)},
  layer.computeNewDeltasForOutputLayer({-1.42857}, {0.5});

  //delta = lossGradients[neuronIndex] * activationDerivative(zPreactivationArray, neuronIndex);
  // = -1.42857*0.5 * (1.0 - 0.5)
  double expectedDelta = -1.42857*0.5 * (1.0 - 0.5);
  double* deltas = layer.getDeltas();
  assert(std::fabs(deltas[0] - expectedDelta) < 1e-6);


  //{ActivationFunction::SIGMOID,
  // z = {0.5},
  // expected =  0.5 * (1.0 - 0.5)},
  layer.computeNewDeltasForHiddenLayer({-1.42857}, {0.5});

  //delta = lossGradients[neuronIndex] * activationDerivative(zPreactivationArray, neuronIndex);
  // = -1.42857*0.5 * (1.0 - 0.5)
  expectedDelta = -1.42857*0.5 * (1.0 - 0.5);
  deltas = layer.getDeltas();
  assert(std::fabs(deltas[0] - expectedDelta) < 1e-6);


  ///Preparing test for propagateDeltas
  double* weights = layer.getWeights();
  double* biases = layer.getBiases();

  // Example weights
  weights[getEntryAndNeuronIndex(0,0,numInputs)] = 0.1;
  weights[getEntryAndNeuronIndex(1,0,numInputs)] = 0.2;
  weights[getEntryAndNeuronIndex(2,0,numInputs)] = 0.3;

  std::vector<double> propagateDeltas = layer.propagateDeltas();
  //+= deltas[neuronIndex] *  weights[neuronIndex * numInputs + entryIndex];
  // expectedDelta * 0.1
  std::vector<double> expectedPropagatedDeltas = {expectedDelta * 0.1, expectedDelta * 0.2, expectedDelta * 0.3};

  assert(vectorsAreEqual(propagateDeltas, expectedPropagatedDeltas, 1e-9));

  std::cout << "All compute New Deltas tests passed!" << std::endl;
}

//TEST OK
void testGenerateGradients() {

  int numInputs = 3;
  int numNeurons = 1;  // Single neuron for simplicity
  ActivationFunction activation = ActivationFunction::SIGMOID; //necessary
  OptimizationAlgorithm optimization = OptimizationAlgorithm::SGD;
  LossFunction loss = LossFunction::CROSS_ENTROPY;

  OptimizedFullyConnectedLayer layer(
      numInputs,
      numNeurons,
      activation,
      loss,
      false,
      optimization
  );
  double* deltas = layer.getDeltas();
  deltas[0] = -0.3571425;

  std::vector<double> input = {1.0, 2.0, 3.0};

  std::vector<double> actualGradietnts =
   layer.computeGradientsWithRespectToWeights(
    input
  );
  //+= deltas[neuronIndex] *   input[entryIndex]
  std::vector<double> expectedGradient = {deltas[0] * 1.0, deltas[0] * 2.0, deltas[0] * 3.0};
  assert(vectorsAreEqual(actualGradietnts, expectedGradient, 1e-9));

  std::cout << "All compute New Gradients tests passed!" << std::endl;
}

//TEST OK
void testUpdateWeightsSGD() {

  int numInputs = 3;
  int numNeurons = 1;  // Single neuron for simplicity
  ActivationFunction activation = ActivationFunction::SIGMOID; //necessary
  OptimizationAlgorithm optimization = OptimizationAlgorithm::SGD;
  LossFunction loss = LossFunction::CROSS_ENTROPY;

  OptimizedFullyConnectedLayer layer(
      numInputs,
      numNeurons,
      activation,
      loss,
      true,
      optimization
  );
  double* weights = layer.getWeights();
  double* biases = layer.getBiases();

  // Example weights
  weights[getEntryAndNeuronIndex(0,0,numInputs)] = 0.1;
  weights[getEntryAndNeuronIndex(1,0,numInputs)] = 0.2;
  weights[getEntryAndNeuronIndex(2,0,numInputs)] = 0.3;

  // Example biases
  biases[0] = 0.1;

  double* deltas = layer.getDeltas();
  deltas[0] = -0.3571425;
  std::vector<double> gradient = {deltas[0] * 1.0, deltas[0] * 2.0, deltas[0] * 3.0};
  layer.updateWeightsSGD(gradient, 0.1);

 // weights[index] -= learningRate * gradients[index]
std::vector<double> newWeights = {
  0.1 - deltas[0] * 0.1,
  0.2 - deltas[0] * 0.2,
  0.3 - deltas[0] * 0.3
 };

  //biases[neuronIndex] -= learningRate * biasGradients[neuronIndex];
  std::vector<double> newBias = { 0.1 - 0.1 * deltas[0] };

  std::cout << "Vector weights : [ ";
  for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex ) {
    for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex ) {
      int entryAndNeuronIndex = (numInputs * neuronIndex) + entryIndex;
      std::cout << weights[entryAndNeuronIndex] << " ";
      assert(std::fabs(weights[entryAndNeuronIndex] - newWeights[entryAndNeuronIndex]) < 1e-6);
    }
  }
  std::cout << "]" << std::endl;

  std::cout << "Vector biases : [ ";
  for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex ) {
      std::cout << biases[neuronIndex] << " ";
      assert(std::fabs(biases[neuronIndex] - newBias[neuronIndex]) < 1e-6);
  }
  std::cout << "]" << std::endl;

  std::cout << "All compute Weiths with SGD tests passed!" << std::endl;

}

void testUpdateWeightsADAM() {

  int numInputs = 3;
  int numNeurons = 1;  // Single neuron for simplicity
  ActivationFunction activation = ActivationFunction::SIGMOID; //necessary
  OptimizationAlgorithm optimization = OptimizationAlgorithm::ADAM;
  LossFunction loss = LossFunction::CROSS_ENTROPY;

  OptimizedFullyConnectedLayer layer(
      numInputs,
      numNeurons,
      activation,
      loss,
      true,
      optimization
  );
  double* weights = layer.getWeights();
  double* biases = layer.getBiases();

  // Example weights
  weights[getEntryAndNeuronIndex(0,0,numInputs)] = 0.1;
  weights[getEntryAndNeuronIndex(1,0,numInputs)] = 0.2;
  weights[getEntryAndNeuronIndex(2,0,numInputs)] = 0.3;

  // Example biases
  biases[0] = 0.1;

  double* deltas = layer.getDeltas();
  deltas[0] = -0.3571425;
  std::vector<double> gradient = {deltas[0] * 1.0, deltas[0] * 2.0, deltas[0] * 3.0};
  layer.updateWeightsAdam(gradient, 0.1, 1);

  // beta1 = 0.9;
  // beta2 = 0.999;
  //epsilon = 1e-8;
//// Calculate bias correction coefficients
// beta1Correction = 1.0 - std::pow(beta1, epoch); = 0.1
// beta2Correction = 1.0 - std::pow(beta2, epoch); = 0.001

//momentum[0] = beta1 * momentum[0] + (1 - beta1) * gradients[0];
//momentum[0] = 0 + 0.1*-0.3571425 = -0.03571425
//momentum[1] = 0 + 0.1*-0.3571425 * 2.0= -0.03571425 * 2.0
//momentum[1] = 0 + 0.1*-0.3571425 * 3.0 = -0.03571425 * 3.0
 //velocity[index] = beta2 * velocity[index] + (1 - beta2) * gradients[index] * gradients[index];
//velocity[0] = 0 + 0.001 * 0.127550765 = 0.000127550765
//velocity[1] = 0 + 0.001 * 0.127550765 * 4 = 0.000127550765 * 4
//velocity[2] = 0 + 0.001 * 0.127550765 * 9 = 0.000127550765 * 9

//mHat[0] = momentum[0] / beta1Correction = -0.3571425
//mHat[1] = -0.3571425 * 2.0
//mHat[2] = -0.3571425 * 3.0
// vHat[0] = velocity[0] / beta2Correction =0.127550765
// vHat[1] =  0.127550765 * 4
// vHat[1] =  0.127550765 * 9

//weights[index] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
//weights[0] -= 0.1 * (-0.3571425) / (0.3571425 + epsilon) = 0.1
//weights[1] -= 0.1 * (-0.3571425)*2 / (0.3571425 *2) = 0.1
//weights[2] -= 0.1 * (-0.3571425)*3 / (0.3571425 *3) = 0.1
std::vector<double> newWeights = {
  0.1 + 0.1,
  0.2 + 0.1,
  0.3 + 0.1
 };

//mBias[0] = beta1 * mBias[0] + (1 - beta1) * biasGradients[0];
//mBias[0] = 0 + (0.1) * (-0.3571425) = -0.03571425
//vBias[0] = beta2 * vBias[0] + (1 - beta2) * biasGradients[0] * biasGradients[0];
//vBias[0] = 0 + (0.001) * 0.127550765 = 0.000127550765

// mHatBias[0] = mBias[0] / beta1Correction;
//mHatBias[0] =-0.3571425
// vHatBias[0] = vBias[0] / beta2Correction;
// vHatBias[0] = 0.127550765

//biases[0] -= learningRate * mHatBias / (sqrt(vHatBias) + epsilon);
//biases[0] =  0.1 - (0.1) * (-0.3571425) / ((0.3571425))

  std::vector<double> newBias = { 0.1 + 0.1 };

  std::cout << "Vector weights : [ ";
  for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex ) {
    for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex ) {
      int entryAndNeuronIndex = (numInputs * neuronIndex) + entryIndex;
      std::cout << weights[entryAndNeuronIndex] << " ";
      assert(std::fabs(weights[entryAndNeuronIndex] - newWeights[entryAndNeuronIndex]) < 1e-6);
    }
  }
  std::cout << "]" << std::endl;

  std::cout << "Vector biases : [ ";
  for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex ) {
      std::cout << biases[neuronIndex] << " ";
      assert(std::fabs(biases[neuronIndex] - newBias[neuronIndex]) < 1e-6);
  }
  std::cout << "]" << std::endl;

  std::cout << "All compute Weiths with ADAM tests passed!" << std::endl;

}

int main() {
    // Run the test
    testPreactivation();
    testActivationDerivative();
    testLossGradients();
    testGenerateDeltas();
    testGenerateGradients();
    testUpdateWeightsSGD();
    testUpdateWeightsADAM();
    return 0;
}
