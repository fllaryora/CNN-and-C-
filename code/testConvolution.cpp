#include <iostream>
#include <vector>
#include "convolution.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // to load images
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // to save images

#include <cassert>
#include <cmath> // Para usar fabs

void assertEqual(const std::vector<std::vector<float>>& vec1, const std::vector<std::vector<float>>& vec2, float epsilon = 1e-5) {
    // Verificar que las dimensiones externas sean iguales
    assert(vec1.size() == vec2.size() && "Vectors have different outer sizes!");

    // Verificar que las dimensiones internas sean iguales
    for (size_t row = 0; row < vec1.size(); ++row) {
        assert(vec1[row].size() == vec2[row].size() && "Vectors have different inner sizes at a specific row!");

        // Verificar que cada valor sea aproximadamente igual (considerando precisión)
        for (size_t col = 0; col < vec1[row].size(); ++col) {
            assert(fabs(vec1[row][col] - vec2[row][col]) <= epsilon && "Vectors have different values!");
        }
    }
}

void printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<float>> convertToGrayscale(unsigned char* image, int width, int height, int channels) {
    std::vector<std::vector<float>> grayscaleImage(height, std::vector<float>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * channels;
            unsigned char r = image[index + 0];
            unsigned char g = image[index + 1];
            unsigned char b = image[index + 2];
            // Convertir a escala de grises usando la fórmula (0.299*R + 0.587*G + 0.114*B)
            float grayValue = 0.299f * r + 0.587f * g + 0.114f * b;
            grayscaleImage[y][x] = grayValue;
        }
    }
    return grayscaleImage;
}

// Función para guardar una imagen en formato PNG
void saveImage(const std::vector<std::vector<float>>& image, const std::string& filename) {
    int width = image[0].size();
    int height = image.size();
    std::vector<unsigned char> outputImage(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            outputImage[y * width + x] = static_cast<unsigned char>(image[y][x]);
        }
    }
    stbi_write_png(filename.c_str(), width, height, 1, outputImage.data(), width);
}


// Convertir imagen cargada a un formato que podamos usar para convolución
std::vector<std::vector<std::vector<float>>> convertToColorChannels(unsigned char* image, int width, int height, int channels) {
    std::vector<std::vector<std::vector<float>>> colorImage(height, std::vector<std::vector<float>>(width, std::vector<float>(channels)));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * channels;
            for (int c = 0; c < channels; ++c) {
                colorImage[y][x][c] = static_cast<float>(image[index + c]);
            }
        }
    }
    return colorImage;
}

// Función para guardar una imagen en color (RGB)
void saveColorImage(const std::vector<std::vector<std::vector<float>>>& image, const std::string& filename) {
    int width = image[0].size();
    int height = image.size();
    int channels = 3;  // Asumimos que siempre son 3 canales (RGB)

    std::vector<unsigned char> outputImage(width * height * channels);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                outputImage[(y * width + x) * channels + c] = static_cast<unsigned char>(std::min(std::max(image[y][x][c], 0.0f), 255.0f));
            }
        }
    }

    stbi_write_png(filename.c_str(), width, height, channels, outputImage.data(), width * channels);
}

// Función para aplicar convolución a una imagen en color (por canales)
std::vector<std::vector<std::vector<float>>> applyConvolutionToColorImage(const std::vector<std::vector<std::vector<float>>>& image, const std::vector<std::vector<float>>& kernel, Convolution& convolution) {
    int height = image.size();
    int width = image[0].size();
    int channels = image[0][0].size();

    std::vector<std::vector<std::vector<float>>> outputImage(height, std::vector<std::vector<float>>(width, std::vector<float>(channels)));

    for (int channel = 0; channel < channels; ++channel) {
        std::vector<std::vector<float>> singleChannel(height, std::vector<float>(width));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                singleChannel[y][x] = image[y][x][channel];
            }
        }

        std::vector<std::vector<float>> convolutedChannel = convolution.applyCrossCorrelation(singleChannel, kernel);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                outputImage[y][x][channel] = convolutedChannel[y][x];
            }
        }
    }

    return outputImage;
}


int main() {
    // Cargar la imagen
    int width, height, channels;
    unsigned char* image = stbi_load("input.png", &width, &height, &channels, 0);
    if (image == nullptr) {
        std::cerr << "Error cargando la imagen\n";
        return -1;
    }

    // Convertir la imagen a escala de grises
    std::vector<std::vector<float>> grayscaleImage = convertToGrayscale(image, width, height, channels);

    // Definir el kernel {1, 0, -1}, {1, 0, -1}, {1, 0, -1}
    std::vector<std::vector<float>> leftSobel = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    };

    std::vector<std::vector<float>> rightSobel = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    std::vector<std::vector<float>> outline = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };

    std::vector<std::vector<float>> blur = {
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f },
        {1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f,1.0f/121.0f }
    };

    // Crear el objeto de convolución
    //Convolution(
       int kernelHeight = 3;
       int kernelWidth = 3;
       int stride = 1;
       int padding = 1; // Usamos padding para mantener el tamaño original
      int poolSize = 2;
      ConvoActFunction activationFunction = RELU;
      ConvoActFunction lineal = LINEAL;
      int kernelBHeight = 11;
      int kernelBWidth = 11;
      int paddingB = 5; // Usamos padding para mantener el tamaño original
    //);
    Convolution convolution(kernelHeight, kernelWidth, stride, padding, poolSize, activationFunction);

    // Aplicar la convolución
    std::vector<std::vector<float>> leftSobelImage = convolution.applyCrossCorrelation(grayscaleImage, leftSobel);

    // Guardar la imagen de salida
    saveImage(leftSobelImage, "leftSobel.png");

    // Aplicar la convolución
    std::vector<std::vector<float>> rightSobelImage = convolution.applyCrossCorrelation(grayscaleImage, rightSobel);

    // Guardar la imagen de salida
    saveImage(rightSobelImage, "rightSobel.png");

    // Aplicar la convolución
    std::vector<std::vector<float>> outlineImage = convolution.applyCrossCorrelation(grayscaleImage, outline);

    // Guardar la imagen de salida
    saveImage(outlineImage, "outline.png");
      ////------------------------Ahora con colores -----
      // Convertir la imagen a canales de color
    std::vector<std::vector<std::vector<float>>> colorImage = convertToColorChannels(image, width, height, channels);

    // Aplicar la convolución a la imagen en color
    std::vector<std::vector<std::vector<float>>> leftSobelColorImage = applyConvolutionToColorImage(colorImage, leftSobel, convolution);

    // Guardar la imagen de salida
    saveColorImage(leftSobelColorImage, "leftSobelColor.png");

    // Aplicar la convolución a la imagen en color
    std::vector<std::vector<std::vector<float>>> rightSobelColorImage = applyConvolutionToColorImage(colorImage, rightSobel, convolution);

    // Guardar la imagen de salida
    saveColorImage(rightSobelColorImage, "rightSobelColor.png");

    // Aplicar la convolución a la imagen en color
    std::vector<std::vector<std::vector<float>>> outlineColorImage = applyConvolutionToColorImage(colorImage, outline, convolution);

    // Guardar la imagen de salida
    saveColorImage(outlineColorImage, "outlineColor.png");

    Convolution soft(kernelBHeight, kernelBWidth, stride, paddingB, poolSize, lineal);
    // Aplicar la convolución a la imagen en color
    std::vector<std::vector<std::vector<float>>> blurImage = applyConvolutionToColorImage(colorImage, blur, soft);

    // Guardar la imagen de salida
    saveColorImage(blurImage, "blur.png");

    // Liberar la memoria de la imagen original
    stbi_image_free(image);

    std::cout << "Convolución aplicada y guardada en 'output.png'\n";

    Convolution test(
      2, 2, //krnel 2x2
       1, //S
        0, //P
        2,// 2x2 Maxpool
         lineal);

    std::vector<std::vector<float>> testMatrix = {
        {10, 25,  1,  5},
        {20, 32, 52,  1},
        { 6,  4,  9, 40},
        {16, 13, 17, 29}
    };

    std::vector<std::vector<float>> testKernel = {
        {1, 0},
        {1, 0}
    };

    std::vector<std::vector<float>> testExpected = {
        {30, 57,53},
        {26, 36, 61},
        {22, 17,26}
    };

    std::vector<std::vector<float>> testConvoExpected = {
        {57,53, 6},
        {36, 61, 41},
        {17,26, 69}
    };

    std::vector<std::vector<float>> maxExpected = {
        {57 }
    };

    std::vector<std::vector<float>> actualResult = test.applyCrossCorrelation(testMatrix, testKernel);
    assertEqual(actualResult, testExpected);
    std::cout << "cross correlation aplicada \n";

    std::vector<std::vector<float>> actualResult2 = test.applyConvolution(testMatrix, testKernel);

    assertEqual(actualResult2, testConvoExpected);
    std::cout << "Convolución aplicada \n";

    std::vector<std::vector<float>> actualResult3 = test.applyMaxPooling(actualResult);
    printMatrix(actualResult3);
    assertEqual(actualResult3, maxExpected);


    std::vector<std::vector<float>> testMatrixMaxPoolExpected = {
        {32,  52},
        { 16,  40}
    };
    std::vector<std::vector<float>> actualResult4 = test.applyMaxPooling(testMatrix);
    printMatrix(actualResult4);
    assertEqual(actualResult4, testMatrixMaxPoolExpected);
    return 0;
}
