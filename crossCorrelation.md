# Convolution and the cross correlation



In mathematics there are several tools that allow us to transform one thing into another thing.
Linear transformation, which is nothing more than a simple matrix multiplication but with a purpose, e.g.
The Quake engine moves the character around the map, each rotation has its matrix, each advance has its matrix.
The Laplace transform is another tool that transforms one thing into another, "but by altering its universe where it resides,
 it takes it to another dimension". 

The laplace transform is heavy and could be optimised into something fast called the FFT discrete fast fourier transform.

This tool transforms a tube full of signals and separates them, and is widely used in sound.
It transforms many signals with amplitude as a function of time into signals with amplitude
as a function of frequency, where these signals are differentiable.

## The reason of the need of a tool of transformation

The artificial neural network is a good tool, but have you ever stopped to think about how the network sees the world?
Humans have millions of skin cells to sense touch, thousands of cones and rods to see the world well,
2 ears that pick up a wide span of frequencies.
The ANN, on the other hand, has few inputs. In other words, we are born with a "great myopia". 

We need a mathematical tool, a pair of mathematical glasses to cure the myopia of the poor ANN.

The idea here is to find a cheap tool to compute mathematical transformation.

Cross correlation is the cheaper version of convolution.

## Convolution

**Convolution** is a mathematical operation on two functions that produces a third function:
One dimention version: (for time for instance).
![fsdf](https://latex.codecogs.com/svg.image?(f\ast&space;g)(t):=\int_{-\infty}^{\infty}f(\tau)g(t-\tau)d\tau)

The term convolution refers to both the result function and to the process of computing it.
An equivalent definition using commutativity is: 
![fsdf](https://latex.codecogs.com/svg.image?(f\ast&space;g)(t):=\int_{-\infty}^{\infty}f(t-\tau)g(\tau)d\tau)

The discrete form is:
![fsdf](https://latex.codecogs.com/svg.image?(f&space;\ast&space;g)[n]=\sum_{m=-\infty}^{\infty}&space;f[m]&space;g[n-m])

using commutativity is: 
![fsdf](https://latex.codecogs.com/svg.image?(f&space;\ast&space;g)[n]=\sum_{m=-\infty}^{\infty}&space;f[n-m]&space;g[m])

### How are the data represented?

A function represents the data stream that IT is going to process to transform and
the function that transforms is a small one, which is called **kernel** of **filter**.

In the case of Deep Learning (Convolutional Neural Networks to be precise), weights of the kernel(a small matrix) are
learnt during training using backpropagation and are not defined/set explicitly.

In order to be able to process a convolution using for programming cycles, the data stream of the functions must be flipped.

**Flippong is a downer** because it makes the things more expensive.


### What is the difference between convolution and cross correlation?

Convolution in CNN involves flipping all the dimensions, for instance 2D both the rows and columns of the kernel before sliding it over the input,
while cross-correlation skips this flipping step.

These operations are foundational in extracting features and detecting patterns within the data, despite their technical differences.

The cross correlation (1 Dimension, time for instance) form is:
![fsdf](https://latex.codecogs.com/svg.image?(f&space;\star&space;g)[n]=\sum_{m=-\infty}^{\infty}&space;f[n+m]&space;g[m])

Images taken from: https://glassboxmedicine.com/2019/07/26/convolution-vs-cross-correlation/
Author: Rachel Draelos, MD, PhD

![image](https://glassboxmedicine.com/wp-content/uploads/2019/07/6_image_expanded.png)

The cross correlation (2 Dimension, 2D gray image for instance) form is:
This is 1 grayscale kernel layer.
For the row i, and column j:

![fsdf](https://latex.codecogs.com/svg.image?(A&space;\star&space;W)[i][j]=\sum_{i'=0}^{k-1}\sum_{j'=0}^{k-1}&space;A[i+i'][j+j']&space;W[i][j])

The cross correlation (3 Dimension, 2D color image for instance) form is:
This is 3 grayscale kernel layer.
For the row i, and column j, C colors:

![fsdf](https://latex.codecogs.com/svg.image?(A&space;\star&space;W)[i][j]=\sum_{i'=0}^{k-1}\sum_{j'=0}^{k-1}\sum_{c=0}^{C-1}&space;A[i+i'][j+j'][c]&space;W[c][i][j])

In most case when you read a pixel from a photo file for each channel (red, blue , green, alpha/transparency ) you get a value between 0 to 255 (integer).
**So, please map this value to 0.0 to 1.0 (float).**
The convolution and cross-correlation, can be seen as a small window that sits on top of the incoming data strem.
Each time it lands on a part of the data stream it generates an output. Then it moves a bit.
A quantity called a **stride** It lands again on another part of the data stream and generates a new output,
and so on until it has covered the full width of the input. 

# A bit of Architecture

In this page: https://github.com/fllaryora/CNN-and-C-/blob/main/layers.md, we talk of the layers of an ANN.
But now if we are talking of a CNN, the ANN layers are the last layers of our CNN.
The CNN layers are the layers in charge of modeling the reality to short-sighted ANN.
So just imagine having a 1920x800 image and a first hidden layer with 100 neurons.
those filters must be very specific to only let pass 100 significative variables.



## The cross correlation as the first layer.

### The change of shape of the layers (Valid, Same and Full)


