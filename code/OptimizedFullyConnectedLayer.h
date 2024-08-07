#ifndef OPTIMIZEDFULLYCONNECTEDLAYER_H
#define OPTIMIZEDFULLYCONNECTEDLAYER_H

#include "FullyConnectedLayer.h"

// Define enums for Optimization Algorithms and Layer Types
enum class OptimizationAlgorithm {
    SGD,
    ADAM
};

class OptimizedFullyConnectedLayer : public FullyConnectedLayer {
public:
    // Constructor
    OptimizedFullyConnectedLayer(
      int numInputs, //1
      int numNeurons, //2
      ActivationFunction activation, //3
      LossFunction loss, //4
      bool useBias, //5
      OptimizationAlgorithm optimization //6
    );

    // Destructor
    ~OptimizedFullyConnectedLayer();

    // Getters for internal state (optional)
    double* getDeltas() const;

    // Output layer
    // $$\frac{\partial \hat{y}_i}{\partial z_i^{(L)}} = \sigma'(z_i^{(L)})$$
    //hidden layer
    //$$\frac{\partial x_i^{(l)}}{\partial z_i^{(l)}} = \sigma'(z_i^{(l)})$$
    double activationDerivative(
      const std::vector<double>& zPreactivationArray,
      int neuronIndex
    ) const;

    std::vector<double> lossDerivative(
      const std::vector<double>& actual,
      const std::vector<double>& expected
    ) const;

   std::vector<double> zPreactivation(
     const std::vector<double>& input
   ) const;

   void computeNewDeltasForOutputLayer(
     const std::vector<double>& lossGradients,
     const std::vector<double>& zPreactivationArray // z_k
   ) const;

   void computeNewDeltasForHiddenLayer(
     const std::vector<double>& propagatedDeltas, // Sum \delta_k^{(l+1)} * W_ki^{(l+1)}
     const std::vector<double>& zPreactivationArray // z_k
   ) const;

   std::vector<double> propagateDeltas() const;

   std::vector<double>
   computeGradientsWithRespectToWeights(
     const std::vector<double>& input // x_k
   ) const;

   void updateWeightsSGD(
     const std::vector<double>& gradients,
     double learningRate) const ;

     void updateWeightsAdam(
       const std::vector<double>& gradients,
       double learningRate,
       int epoch
     ) const;

     void backwardForOutputLayer(
       const std::vector<double>& input,
        const std::vector<double>& expected,
         double learningRate, int epoch
       ) const;

       void backwardForHiddenLayer(
         const std::vector<double>& input,
          const std::vector<double>& propagatedDeltas, // Sum \delta_k^{(l+1)} * W_ki^{(l+1)}
        double learningRate,
        int epoch
         ) const;
private:
    OptimizationAlgorithm optimizationAlgorithm;

    double* momentum;      // First moment vector for Adam
    double* velocity;      // Second moment vector (analogous to variance) for Adam
    double* mBias;  // Bias momentum vector for Adam
    double* vBias;  // Bias velocity vector for Adam

    double* deltas; // Delta errors for each neuron

    // Hyperparameters that control the decay rates for
    // the first and second moments, respectively.
    double beta1 = 0.9;  // Adam parameter
    double beta2 = 0.999; // Adam parameter
    double epsilon = 1e-8; // Adam parameter like pytorch

};

#endif // OPTIMIZEDFULLYCONNECTEDLAYER_H
