#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

enum ConvoActFunction {
    LINEAL,
    RELU,
    LEAKY_RELU,
    GELU,
};

class Convolution {
public:
    // Constructor
    Convolution(int kernelHeight,
       int kernelWidth,
       int stride,
       int padding,
        int poolSize,
        ConvoActFunction activationFunction);

    // Main methods
    std::vector<std::vector<float>> applyConvolution(
      const std::vector<std::vector<float>>& inputData,
      const std::vector<std::vector<float>>& kernel);
    std::vector<std::vector<float>> applyCrossCorrelation(
      const std::vector<std::vector<float>>& inputData,
      const std::vector<std::vector<float>>& kernel);
  const float applyActivationFunction(
      const float inputData);
    std::vector<std::vector<float>> applyMaxPooling(
      const std::vector<std::vector<float>>& inputData);

private:
    int kernelHeight;
    int kernelWidth;
    int stride;
    int padding;
    int poolSize;
    ConvoActFunction activationFunction;

    // Helper methods
    std::vector<std::vector<float>> applyOperation(
      const std::vector<std::vector<float>>& inputData,
      const std::vector<std::vector<float>>& kernel, bool isConvolution);
    bool isPaddingPosition(int dataHeightIndex,
      int dataWidthIndex, int paddedHeight, int paddedWidth);
    float applyReLU(float value);
    float applyLeakyReLU(float value);
    float applyGELU(float value);
    float getMaxInRegion(const std::vector<std::vector<float>>& region);
};

#endif // CONVOLUTION_H
