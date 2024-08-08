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

### How are the data represented?

A function represents the data stream that IT is going to process to transform and
the function that transforms is a small one, which is called a kernel.

In the case of Deep Learning (Convolutional Neural Networks to be precise), weights of the kernel(a small matrix) are
learnt during training using backpropagation and are not defined/set explicitly.

In order to be able to process a convolution using for programming cycles, the data stream of the functions must be flipped.

**Flippong is a downer** because it makes the things more expensive.

### What is the difference between convolution and cross correlation?

Convolution in CNN involves flipping all the dimensions, for instance 2D both the rows and columns of the kernel before sliding it over the input,
while cross-correlation skips this flipping step.

These operations are foundational in extracting features and detecting patterns within the data, despite their technical differences.


