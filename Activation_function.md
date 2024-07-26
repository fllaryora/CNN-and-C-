### No activation function case (Linear regression case)

OK, but what do I mean by this? Well, **first**, let's suppose that we have a number of neurons that are chained together and that there is no activation function.
So, the equation of the output of the last neuron will take the form of:

https://latex.codecogs.com/svg.image?a_i=b&plus;w_0&space;x_0&plus;w_1&plus;x_1&plus;...&plus;w_m&space;x_m

$$ a_i = b + w_0 x_0 + w_1 + x_1 + ...+ w_m x_m $$
Credits to wikipedia:
<img src="Linear_regression.svg" alt="Image of a neuron" style="height: 400px; width:400px;"/>

So, if we want to predict a linear regression without the classic equation:

https://latex.codecogs.com/svg.image?b_1=\frac{\sum_{i=0}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=0}^{n}(x_i-\bar{x})^{2}}

https://latex.codecogs.com/svg.image?b_0=\bar{y}-(b_1\bar{x})

https://latex.codecogs.com/svg.image?y=b_0&plus;(b_1&space;x)

$$b_1 = \frac{\sum\limits_{i=0}^n  (x_i-\bar{x})(y_i-\bar{y})  }{\sum\limits_{i=0}^n  (x_i-\bar{x})^{2} }$$

$$b_0 = \bar{y}-(b_1\bar{x})$$

$$ y = b_0 = b_1 x $$

**Therefore, An activation function will exists if we want to do better things. Other wise our network will be limited to perform only a linear regression.**

### Binary Classification case (Logistic regression case)

Now, **second**, let's suppose that we have set of incomes and we want to classify them by type red and blue:
With a linear activation function we collapse again in a linear regression case:

<img src="linear_train.png" alt="Image of a neuron" style="height: 400px; width:400px;"/>

And it will be impossible with a network of neurons to classify things like this:

<img src="non_linear_train.png" alt="Image of a neuron" style="height: 400px; width:400px;"/>

Image source: https://cfml.se/blog/binary_classification/

- **First** because of the curve of the function is a line.
- However, so far I have not talked at all about learning of neural networks with backpropagation.
 The **second** reason which you can find, when you try to train the neuron:
  the the setting of the neuron weights will be the same while you apply backpropagation, no matter what the input is.
  In other words, without knowing backpropagation method, you simply cannot use the activation function as a universal approximation function.

**Therefore, An activation function will be non-linear if we want to do better things. Again, other wise our network will be limited to perform only a linear clasification.**

### The activation function must be continuously difeerentiable

So far, We talked about neurons and their weights, and we talked about them as if they were ideals and we set them from the beginning.
But this is not the case. At the beginning, the weights are set with random values. So, our network goes through a learning process at some point.
In the case of backpropagation, it is based on the stochastic gradient descent (SGD) method.

Gradient descent was originally proposed by CAUCHY in 1847. It is also known as the steepest descent.

<img src="SGD.png" alt="Image of a neuron" />

Here I am suppoussing we have a single neuron:

$$ w_t = w_{t-1} - \eta \nabla{\varphi (a_{i})} $$

Where **eta** is called **learning rate**.
About the notation used for learning rate: In other places you might find that they use the Greek letter **alpha** instead of eta.
About the notation used for weights: In other places you might find that they use the Greek letter **theta** instead of w.

And nabla is the simbol of gradient this represenst:

https://latex.codecogs.com/svg.image?\nabla&space;f(x_1,...,x_n)={\begin{bmatrix}{\frac{\partial&space;f}{\partial&space;x_1}}(x_1,...,x_n)\\\vdots\\{\frac{\partial&space;f}{\partial&space;x_n}}(x_1,...,x_n)\end{bmatrix}}

$$\ 
\nabla f(x_1, \ldots, x_n) = \begin{bmatrix} 
\frac{\partial f}{\partial x_1}(x_1, \ldots, x_n) \\ 
\vdots \\ 
\frac{\partial f}{\partial x_n}(x_1, \ldots, x_n) 
\end{bmatrix} 
\$$

As I am suppoussing we have a single neuron, the gradien is the first derivative of phi.

$$ w_t = w_{t-1} - \eta \varphi'(a) $$

It is time to give 2 examples of activation functions used: the logistic function or **sigmoidal** function and the **hyperbolic tangent** function.

#### sigmoidal

$${\displaystyle \sigma (x)\doteq {\frac {1}{1+e^{-x}}}}$$

$${\displaystyle \sigma'(x)\doteq \sigma(x)(1-\sigma(x))}$$

<img src="Activation_logistic.svg" alt="Image of a neuron" />

#### hyperbolic tangent

$${\displaystyle \tanh(x)\doteq {\frac {e^{x}-e^{-x}}{e^{x}+e^{-x}}}}$$

$${\displaystyle \tanh(x)\doteq 1-tanh(x)^{2}}$$

<img src="Activation_tanh.svg" alt="Image of a neuron"/>

**So, The activation function must be continuously differentiable. But, being derivable is not the only problem that activation functions have to face.**

### The range of activation function should be infinite.

Activation functions such as sigmoidal and tanh tend to cause the **problem of vanishing gradient** during deep network training.
This is because their gradients approach zero as input values become large or small.
And if the their gradients approach zero, then next weight won't be modified efficiently.

So, It is time to present you the ReLU activation function's family:

https://latex.codecogs.com/svg.image?ReLU(x)={\begin{cases}0&{\text{if}}x\leq&space;0\\x&{\text{if}}x>0\end{cases}}

$$ReLU(x)={\begin{cases}0&{\text{if }}x\leq 0\\
x&{\text{if }}x>0\end{cases}}$$

$$ReLU'(x)={\begin{cases}0&{\text{if }}x\leq 0\\
1&{\text{if }}x>0\end{cases}}$$

Why, if we said that the activation function is continuously differentiable,
do we propose as an improvement a function that is not differentiable at zero?
For many reasons, it is differentiable everywhere else, and the value of the derivative at zero can be arbitrarily chosen to be 0 or 1.
It has efficient computation: Only comparison, addition and multiplication.
**Sparse activation**: For example, in a randomly initialized network, only about 50% of hidden units are activated (have a non-zero output)
Better **gradient propagation**: Fewer vanishing gradient problems compared to sigmoidal activation functions that saturate in both directions.

- **Not zero-centered**: ReLU outputs are always non-negative. This can make it harder for the network to learn during backpropagation because gradient updates tend to push weights in one direction (positive or negative). So here you have the leaky relu, ELU and SoftPlus.
  
$$Leaky ReLU(x)={\begin{cases}0.01x&{\text{if }}x\leq 0\\
x&{\text{if }}x>0\end{cases}}$$

$$Leaky ReLU'(x)={\begin{cases}0.01&{\text{if }}x\leq 0\\
1&{\text{if }}x>0\end{cases}}$$

$$Softplus(x)=ln(1+e^{x})$$

$$Softplus'(x)=\frac {1}{1+e^{-x}}$$

$$ELU(x,\alpha)={\begin{cases}\alpha(e^{x}-1)&{\text{if }}x\leq 0\\
x&{\text{if }}x>0\end{cases}}$$

$$ELU'(x)={\begin{cases}\alpha(e^{x})&{\text{if }}x\leq 0\\
1&{\text{if }}x>0\end{cases}}$$

# Folding activation functions

These functions are applied collectively to the vk outputs of all neurons in a layer, transforming them into a coherent set of outputs.

## Softmax
 Softmax takes the outputs (vk) of all the neurons in the layer and converts them into probabilities that sum to 1. It is commonly used in the output layer of multi-class classifiers.
For all the neurons in the layer, vk or z in other notation, from the neuron 1, ...to j

https://latex.codecogs.com/svg.image?softmax(vk)_i=\frac{e^{vk_i}}{\sum_{j=1}^{K}e^{vk_j}}

![softmax](https://latex.codecogs.com/png.latex?\sigma(\mathbf{vk})_i=\frac{e^{vk_i}}{\sum_{j=1}^{K}e^{vk_j}})

![softmax](https://latex.codecogs.com/svg.image?softmax'(\vec{vk})_i=\frac{\partial&space;softmax\left({\vec{vk}}\right)_i}{\partial&space;x_{j}}=softmax_{i}\left({\vec{x}}\right)\left(\delta&space;_{ij}-softmax_{j}\left({\vec{x}}\right)\right))

$$ \delta _{ij}={\begin{cases}0&{\text{if }}i\neq j,\\
1&{\text{if }}i=j.\end{cases}} $$
