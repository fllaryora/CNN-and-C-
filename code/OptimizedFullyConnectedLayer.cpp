#include "OptimizedFullyConnectedLayer.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Constructor
OptimizedFullyConnectedLayer::OptimizedFullyConnectedLayer(
  int numInputs, //1
  int numNeurons, //2
  ActivationFunction activation,
  LossFunction loss, //4
  bool useBias, //5
  OptimizationAlgorithm optimization //6
) : FullyConnectedLayer(numInputs, numNeurons, activation, loss, useBias),
    optimizationAlgorithm(optimization) {

    // Initialize arrays for Adam optimization if necessary
    if (optimizationAlgorithm == OptimizationAlgorithm::ADAM) {

        momentum = new double[numInputs * numNeurons]();
        velocity = new double[numInputs * numNeurons]();
        for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
            for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex) {
                int index = neuronIndex * numInputs + entryIndex;
                // set 0 for t = 0
                momentum[index] = 0.0;
                velocity[index] = 0.0;
            }
        }
        if (useBias) {
            mBias = new double[numNeurons]();
            vBias = new double[numNeurons]();
            for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
              // set 0 for t = 0
                mBias[neuronIndex] = 0.0;
                vBias[neuronIndex] = 0.0;
            }
        } else {
            mBias = nullptr;
            vBias = nullptr;
        }
    } else {
        momentum = nullptr;
        velocity = nullptr;
        mBias = nullptr;
        vBias = nullptr;
    }

    deltas = new double[numNeurons](); // Initialize delta errors
}

// Destructor
OptimizedFullyConnectedLayer::~OptimizedFullyConnectedLayer() {
    delete[] deltas;

    if (momentum) {
        delete[] momentum;
        delete[] velocity;
    }

    if (mBias) {
        delete[] mBias;
        delete[] vBias;
    }
}

// Getters (optional) //OK
double* OptimizedFullyConnectedLayer::getDeltas() const {
    return deltas;
}


// Output layer
// $$\frac{\partial \hat{y}_i}{\partial z_i^{(L)}} = \sigma'(z_i^{(L)})$$
//hidden layer
//$$\frac{\partial x_i^{(l)}}{\partial z_i^{(l)}} = \sigma'(z_i^{(l)})$$
//TESTED OK
double OptimizedFullyConnectedLayer::activationDerivative(
  const std::vector<double>& zPreactivationArray,
  int neuronIndex
) const {
  if (zPreactivationArray.size() != numNeurons) {
      std::cerr << "Error: Size of zPreactivationArray and numNeurons must match.\n";
      return -1.0;
  }
    switch (activationFunction) {
        case ActivationFunction::RELU:
            return zPreactivationArray[neuronIndex] > 0.0 ? 1.0 : 0.0;
        case ActivationFunction::SIGMOID:
            return zPreactivationArray[neuronIndex] *
             (1.0 - zPreactivationArray[neuronIndex]);
        case ActivationFunction::TANH:
            return 1.0 -
             std::tanh(zPreactivationArray[neuronIndex]) *
              std::tanh(zPreactivationArray[neuronIndex]);
        case ActivationFunction::SOFTMAX:
        {
          std::vector<double> softmaxArray = softmax(zPreactivationArray);
          return softmaxArray[neuronIndex] *
           (1.0 - softmaxArray[neuronIndex]);
        }
        case ActivationFunction::LINEAR:
        default:
            return 1.0; // Default for linear activation
    }
}

// Loss function derivative
/**
//MSE The theoric equation is
 $$\frac{\partial \text{MSE}}{\partial \hat{y}_k^{(L)}} =
     \frac{-2}{n} (y_k - \hat{y}_k )$$

 Removing constants simplifies the code and makes it computationally
 more efficient without significantly affecting the convergence or learning process.
 So $$\frac{\partial \text{MSE}}{\partial \hat{y}_k^{(L)}} = \hat{y}_k - y_k $$
 The constant factor 2 and division by n (number of neurons or samples)
  can be absorbed into the learning rate.
  This means you can adjust the learning rate hyperparameter to account
   for these scaling factors, making them unnecessary in the derivative calculation.

//BCE The theoric equation is
 $$\frac{\partial \text{BCE}}{\partial \hat{y}_k^{(L)}} =
    \frac{-1}{K} ( \frac{y_k}{\hat{y}_k} - \frac{1-y_k}{1-\hat{y}_k} )$$
Removing constants
$$\frac{\partial \text{BCE}}{\partial \hat{y}_k^{(L)}} =
   -( \frac{y_k}{\hat{y}_k} - \frac{1-y_k}{1-\hat{y}_k} )$$
//CCE
//  $$\frac{\partial \text{CCE}}{\partial \hat{y}_k^{(L)}} =
//     - \frac{y_k}{\hat{y}_k}$$
*/
std::vector<double> OptimizedFullyConnectedLayer::lossDerivative(
    const std::vector<double>& actual, //  \hat{y}_k
    const std::vector<double>& expected //  y_k
) const { //TESTED OK
  if (actual.size() != numNeurons) {
      std::cerr << "Error: Size of actual and numNeurons must match.\n";
      std::vector<double> lossGradients(numNeurons, -1.0);
      return lossGradients;
  }
  if (expected.size() != numNeurons) {
      std::cerr << "Error: Size of expected and numNeurons must match.\n";
      std::vector<double> lossGradients(numNeurons, -1.0);
      return lossGradients;
  }
  std::vector<double> lossGradients(numNeurons);

    switch (lossFunction) {
        case LossFunction::MSE:
            for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
              // lossGradients[neuronIndex] = -2 * (expected[neuronIndex] - actual[neuronIndex]) / numNeurons;
              lossGradients[neuronIndex] = actual[neuronIndex] - expected[neuronIndex];
            }
            break;

        case LossFunction::CROSS_ENTROPY:
            for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
              lossGradients[neuronIndex] = -(
                actual[neuronIndex] / (expected[neuronIndex] + 1e-9)
                - (1 - actual[neuronIndex]) / (1 - expected[neuronIndex] + 1e-9)
              );
            }
            break;

        case LossFunction::CATEGORICAL_CROSS_ENTROPY:
            // prevent zero dev with + 1e-9
            for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
                lossGradients[neuronIndex] = -expected[neuronIndex] / (actual[neuronIndex] + 1e-9);
            }
            break;
    }

    return lossGradients;
}

//TESTED OK
std::vector<double> OptimizedFullyConnectedLayer::zPreactivation(
  const std::vector<double>& input
) const {
  if (input.size() != numInputs) {
      std::cerr << "Error: Size of input and numInputs must match.\n";
      std::vector<double> zPreactivationArray(numNeurons, -1.0);
      return zPreactivationArray;
  }
  std::vector<double> zPreactivationArray(numNeurons, 0.0);
  for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
      double zPreactivation = 0.0;
      for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex) {
        // the max Input EntryAmount In Order To Jump the next neuron is numInputs
          int entryAndNeuronIndex = (numInputs * neuronIndex) + entryIndex;
          zPreactivation += input[entryIndex] * weights[ entryAndNeuronIndex ];
      }
      if (useBias) {
          zPreactivation += biases[neuronIndex];
      }
      zPreactivationArray[neuronIndex] = zPreactivation;
  }
  return zPreactivationArray;
}

//One delta per neuron.
void OptimizedFullyConnectedLayer::computeNewDeltasForOutputLayer(
  const std::vector<double>& lossGradients,
  const std::vector<double>& zPreactivationArray // z_k
) const { //TESTED OK
  if (lossGradients.size() != numNeurons) {
      std::cerr << "Error: Size of lossGradients and numNeurons must match.\n";
      return ;
  }
  for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
    deltas[neuronIndex] = lossGradients[neuronIndex] *
     activationDerivative(zPreactivationArray, neuronIndex);
  }
}


//One delta per neuron.
void OptimizedFullyConnectedLayer::computeNewDeltasForHiddenLayer(
  const std::vector<double>& propagatedDeltas, // Sum \delta_k^{(l+1)} * W_ki^{(l+1)}
  const std::vector<double>& zPreactivationArray // z_k
) const { //TESTED OK
 //One propagatedDeltas per neuron.
 if (propagatedDeltas.size() != numNeurons) {
     std::cerr << "Error: Size of propagatedDeltas and numNeurons must match.\n";
     return ;
 }
  for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
    deltas[neuronIndex] = propagatedDeltas[neuronIndex] *
     activationDerivative(zPreactivationArray, neuronIndex);
  }
}
// the last function is equivalent to say:
// computeGradientsWithRespectToBiasForHiddenLayer

// Propagate deltas to the previous layer
//Once the owning deltas are calculated
std::vector<double> OptimizedFullyConnectedLayer::propagateDeltas(

) const { //TEST OK
  //One propagatedDeltas per neuron in the previous layer.
    std::vector<double> propagatedDeltas(numInputs, 0.0);

    for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex) {
        for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
            propagatedDeltas[entryIndex] += deltas[neuronIndex] *
              weights[neuronIndex * numInputs + entryIndex];
        }
    }
    return propagatedDeltas;
}


// Compute gradients
// \frac{\partial\text{LOSS}}{\partial&space;W_{jk}^{(L)}}=
//    \delta_k^{(L)}\cdot\frac{\partial&space;z_k}{\partial&space;W_{jk}^{(L)}}=
//    \delta_k^{(L)}\cdot&space;x_{j}^{(L-1)}
std::vector<double>
OptimizedFullyConnectedLayer::computeGradientsWithRespectToWeights(
  const std::vector<double>& input // x_k
) const { //TEST OK
  if (input.size() != numInputs) {
      std::cerr << "Error: Size of input and numInputs must match.\n";
      std::vector<double> gradients(numInputs * numNeurons, -1.0);
      return gradients;
  }
  //one gradient per owning weight
  std::vector<double> gradients(numInputs * numNeurons, 0.0);

  for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
      for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex) {
          int index = neuronIndex * numInputs + entryIndex;
          gradients[index] = deltas[neuronIndex] * input[entryIndex];
      }
  }

  return gradients;
}


// Update weights using SGD
void OptimizedFullyConnectedLayer::updateWeightsSGD(
  const std::vector<double>& gradients,
  double learningRate) const { //TEST OK
    const double* biasGradients = deltas;
    for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
        for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex) {
            int index = neuronIndex * numInputs + entryIndex;
            weights[index] -= learningRate * gradients[index];
        }
        if (useBias) {
            biases[neuronIndex] -= learningRate * biasGradients[neuronIndex];
        }
    }
}

// Update weights using Adam optimization
// The **first moment** (or momentum)
// $$m_t=(\beta_1 \cdot m_{t-1}) + (1-\beta_1) \cdot \frac{\partial L}{\partial W^{(l)}_{jk(t-1)}}$$
//The **second moment** (or velocity)
// $$v_t=(\beta_2 \cdot v_{t-1}) + (1-\beta_2) \cdot (\frac{\partial L}{\partial W^{(l)}_{jk(t-1)}})^2$$
// \hat{m}_t=\frac{m_{t-1}}{1-\beta_1^{t}}
//\hat{v}_t=\frac{v_{t-1}}{1-\beta_2^{t}}

void OptimizedFullyConnectedLayer::updateWeightsAdam(
  const std::vector<double>& gradients,
  double learningRate,
  int epoch
) const { //TEST OK
  const double* biasGradients = deltas;

  // Calculate bias correction coefficients
  double beta1Correction = 1.0 - std::pow(beta1, epoch);
  double beta2Correction = 1.0 - std::pow(beta2, epoch);

    for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) {
        for (int entryIndex = 0; entryIndex < numInputs; ++entryIndex) {
            int index = neuronIndex * numInputs + entryIndex;

            // Update biased first moment estimate
            momentum[index] = beta1 * momentum[index] + (1 - beta1) * gradients[index];
            // Update biased second raw moment estimate
            velocity[index] = beta2 * velocity[index] + (1 - beta2) * gradients[index] * gradients[index];

            // Compute bias-corrected first and second moment estimates
            double mHat = momentum[index] / beta1Correction;
            double vHat = velocity[index] / beta2Correction;

            // Update weights
            weights[index] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
        }

        if (useBias) {
            // Update biased first moment estimate for bias
            mBias[neuronIndex] = beta1 * mBias[neuronIndex] + (1 - beta1) * biasGradients[neuronIndex];
            // Update biased second raw moment estimate for bias
            vBias[neuronIndex] = beta2 * vBias[neuronIndex] + (1 - beta2) * biasGradients[neuronIndex] * biasGradients[neuronIndex];

            // Compute bias-corrected estimates for biases
            double mHatBias = mBias[neuronIndex] / beta1Correction;
            double vHatBias = vBias[neuronIndex] / beta2Correction;
            // Update biases
            biases[neuronIndex] -= learningRate * mHatBias / (sqrt(vHatBias) + epsilon);
        }
    }
}

void OptimizedFullyConnectedLayer::backwardForOutputLayer(
  const std::vector<double>& input,
   const std::vector<double>& expected,
    double learningRate, int epoch
  ) const {
    std::vector<double> zPreactivationArray = zPreactivation(input);
    std::vector<double> actual = forward(input);
    std::vector<double> lossGradients = lossDerivative(actual, expected);
    computeNewDeltasForOutputLayer(lossGradients, zPreactivationArray);
    std::vector<double> gradients =
     computeGradientsWithRespectToWeights(input);

    if(OptimizationAlgorithm::SGD == optimizationAlgorithm){
      updateWeightsSGD(gradients,learningRate);
    } else {
      updateWeightsAdam(gradients,learningRate, epoch);
    }
}

void OptimizedFullyConnectedLayer::backwardForHiddenLayer(
  const std::vector<double>& input,
   const std::vector<double>& propagatedDeltas, // Sum \delta_k^{(l+1)} * W_ki^{(l+1)}
    double learningRate, int epoch
  ) const {

    std::vector<double> actual = forward(input);

    // Sum \delta_k^{(l+1)} * W_ki^{(l+1)}
    computeNewDeltasForHiddenLayer(input, propagatedDeltas);
    std::vector<double> gradients =
     computeGradientsWithRespectToWeights(input);

    if(OptimizationAlgorithm::SGD == optimizationAlgorithm){
      updateWeightsSGD(gradients,learningRate);
    } else {
      updateWeightsAdam(gradients,learningRate, epoch);
    }
}
