#include "convolution.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// Constructor
Convolution::Convolution(
  int kernelHeight,
  int kernelWidth,
  int stride,
  int padding,
  int poolSize,
  ConvoActFunction activationFunction
) : kernelHeight(kernelHeight),
kernelWidth(kernelWidth),
stride(stride),
padding(padding),
poolSize(poolSize),
activationFunction(activationFunction) {

}

// Apply convolution
std::vector<std::vector<float>> Convolution::applyConvolution(
  const std::vector<std::vector<float>>& inputData,
  const std::vector<std::vector<float>>& kernel
) {
    return applyOperation(inputData, kernel, true);
}

// Apply cross-correlation
std::vector<std::vector<float>> Convolution::applyCrossCorrelation(
  const std::vector<std::vector<float>>& inputData,
  const std::vector<std::vector<float>>& kernel) {
    return applyOperation(inputData, kernel, false);
}

// Helper function to perform convolution or cross-correlation
std::vector<std::vector<float>> Convolution::applyOperation(
  const std::vector<std::vector<float>>& inputData,
  const std::vector<std::vector<float>>& kernel, bool isConvolution) {
    int paddedHeight = inputData.size() + 2 * padding;
    int paddedWidth = inputData[0].size() + 2 * padding;
    int outputHeight = (paddedHeight - kernelHeight) / stride + 1;
    int outputWidth = (paddedWidth - kernelWidth) / stride + 1;

    std::vector<std::vector<float>> outputData(
      outputHeight,
      std::vector<float>(outputWidth)
    );

    for (int outputHeightIndex = 0; outputHeightIndex < outputHeight; ++outputHeightIndex) {
        int startKernelHeightIndex = std::max(
          0,
           padding - outputHeightIndex * stride
         );
        int endKernelHeightIndex = std::min(
          kernelHeight,
          static_cast<int>( inputData.size() ) + padding - outputHeightIndex * stride
        );
        for (int outputWidthIndex = 0; outputWidthIndex < outputWidth; ++outputWidthIndex) {
            float value = 0.0f;
            // Calculate valid kernel index range based on padding and stride
            int startKernelWidthIndex = std::max(
              0,
              padding - outputWidthIndex * stride
            );
            int endKernelWidthIndex = std::min(
              kernelWidth,
              static_cast<int>( inputData[0].size() ) + padding - outputWidthIndex * stride
            );

            for (
              int kernelHeightIndex = startKernelHeightIndex;
              kernelHeightIndex < endKernelHeightIndex;
              ++kernelHeightIndex) {

                int inputHeightIndex = outputHeightIndex * stride + kernelHeightIndex - padding;

                for (
                  int kernelWidthIndex = startKernelWidthIndex;
                  kernelWidthIndex < endKernelWidthIndex;
                  ++kernelWidthIndex) {

                    int inputWidthIndex = outputWidthIndex * stride + kernelWidthIndex - padding;

                    float kernelValue = isConvolution
                        ? kernel[kernelHeight - kernelHeightIndex - 1][kernelWidth - kernelWidthIndex - 1]
                        : kernel[kernelHeightIndex][kernelWidthIndex];

                    value += inputData[inputHeightIndex][inputWidthIndex] * kernelValue;
                }
            }

            outputData[outputHeightIndex][outputWidthIndex] = applyActivationFunction(value);
        }
    }

    return outputData;
}

// Apply activation function
const float Convolution::applyActivationFunction(
  const float inputData
) {
      if (activationFunction == RELU) {
          return applyReLU(inputData);
      } else if (activationFunction == LEAKY_RELU) {
          return applyLeakyReLU(inputData);
      } else if (activationFunction == GELU) {
          return applyGELU(inputData);
      }
      return inputData;
}

// ReLU activation
float Convolution::applyReLU(float value) {
    return std::max(0.0f, value);
}

// Leaky ReLU activation
float Convolution::applyLeakyReLU(float value) {
    return (value > 0) ? value : 0.01f * value;
}

// GELU activation (approximation)
float Convolution::applyGELU(float value) {
    return 0.5f * value * (1.0f + std::tanh(sqrt(2.0f / M_PI) *
     (value + 0.044715f * pow(value, 3))));
}

// Apply max pooling
std::vector<std::vector<float>> Convolution::applyMaxPooling(
  const std::vector<std::vector<float>>& inputData
) {
    int outputHeight = inputData.size() / poolSize;
    int outputWidth = inputData[0].size() / poolSize;
    std::vector<std::vector<float>> outputData(
      outputHeight, std::vector<float>(outputWidth)
    );

    for (int outputHeightIndex = 0; outputHeightIndex < outputHeight; ++outputHeightIndex) {
        for (int outputWidthIndex = 0; outputWidthIndex < outputWidth; ++outputWidthIndex) {

          int inputHeightIndex = outputHeightIndex * poolSize ;
          int inputWidthIndex = outputWidthIndex * poolSize ;

            float maxValue = inputData[inputHeightIndex][inputWidthIndex];

            for (int poolHeightIndex = 0; poolHeightIndex < poolSize; ++poolHeightIndex) {
                for (int poolWidthIndex = 0; poolWidthIndex < poolSize; ++poolWidthIndex) {
                    int inputHeightIndex = outputHeightIndex * poolSize + poolHeightIndex;
                    int inputWidthIndex = outputWidthIndex * poolSize + poolWidthIndex;

                    float possibleValue = inputData[inputHeightIndex][inputWidthIndex];
                    if(possibleValue > maxValue){
                      maxValue = possibleValue;
                    }
                }
            }
            outputData[outputHeightIndex][outputWidthIndex] = maxValue;
        }
    }

    return outputData;
}
