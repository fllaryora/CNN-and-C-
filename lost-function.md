# List of error calculus functions and their derivatives. 
These functions give the error curve to be applied to the backpropagation algorithm. To search for a local or global minimum with luck. 

## Mean Square Error method for regression

 $$MSE(\vec{y},\vec{\hat{y}})=\frac{1}{K}\sum_{k=1}^{K}( y_i - \hat{y}_i)^{2}$$
 
Old ANN use it even for classification but it is not the best for this purpose.
This method is based on the gausian distribution (bell curve).
MSE is the value for the entire output layer, it is not the individual error for each neuron.

Let the weight from the layer l-1 neuron i to layer l neuron j

$$W_{ij}^{l}$$

We want to calculate each ij

$$\frac{\partial MSE}{\partial W_{ij}^{l}}$$

the rate of change of the lost error function with respect to the given connective weight, so we can minimize it.
Now we consider two cases, the output of the network and the output of some middle hidden layer.

Notation

$$W_{jk}$$​: Element at row j(neurons), column k (entries) of the weight matrix W.

$$X_{ik}$$: Element at row i(samples), column k (entries)of the input matrix X.

$$z_i = \sum_{k=1}^{m} X_{ik}W_{jk} + b_j$$​: Linear combination for the i-th sample.

$$\hat{y}_i = f(z_i)$$​: Predicted output for the i-th sample.

$$y_i$$: True label for the i-th sample.

f: Activation function.

f′: Derivative of the activation function.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ : the Mean Square Error (MSE)

**first** Derivative of MSE with Respect to $$\hat{y}_i$$

$$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n} (\hat{y}_i - y_i)$$

**Second**  Derivative of $$\hat{y}_i$$ with Respect to $$z_i$$

$$\frac{\partial \hat{y}_i}{\partial z_i} = f'(z_i)$$

(we saw examples of this in the activation function page)

**third** this is the harder step for my brain: Derivative of $$z_i$$​ with Respect to $$W_{jk}$$​:

$$\frac{\partial z_i}{\partial W_{jk}} = X_{ij}$$

**fourth** Chain Rule for Partial Derivative of MSE with Respect to $$W_{jk}$$​:

![ahh](https://latex.codecogs.com/svg.image?\frac{\partial\text{MSE}}{\partial&space;W_{jk}}=\frac{1}{n}\sum_{i=1}^{n}\frac{\partial\text{MSE}}{\partial\hat{y}_i}\cdot\frac{\partial\hat{y}_i}{\partial&space;z_i}\cdot\frac{\partial&space;z_i}{\partial&space;W_{jk}})

**fifth** Substituting the derivatives:

![ahh](https://latex.codecogs.com/svg.image?\frac{\partial\text{MSE}}{\partial&space;W_{jk}}=\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)\cdot&space;f'(z_i)\cdot&space;X_{ij})

**6**Derivative of $$z_i$$​ with Respect to b (that layer has j neurons)

$$\frac{\partial z_i}{\partial b_j} = 1$$

**7** Chain Rule for Partial Derivative of MSE with Respect to b

![ahh](https://latex.codecogs.com/svg.image?\frac{\partial\text{MSE}}{\partial&space;b_j}=\frac{1}{n}\sum_{i=1}^{n}\frac{\partial\text{MSE}}{\partial\hat{y}_i}\cdot\frac{\partial\hat{y}_i}{\partial&space;z_i}\cdot\frac{\partial&space;z_i}{\partial&space;b_j})


**8** Substituting the derivatives:

![ahh](https://latex.codecogs.com/svg.image?\frac{\partial\text{MSE}}{\partial&space;b_j}=\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)\cdot&space;f'(z_i))

## Binary Cross Entropy method for Classification

![Bin](https://latex.codecogs.com/svg.image?&space;BinaryCrossEntropy(\vec{y},\vec{\hat{y}})=\frac{1}{K}\sum_{k=1}^{K}{y_i\log_{e}\hat{y}_i&plus;(1-y_i)\log_{e}(1-y_i)})

This method is based on the bernoulli distribution .
Binary Cross Entropy is the value for the entire output layer, it is not the individual error for each neuron.

## Categorical Cross Entropy method for Classification must end only in one category 

![Cat](https://latex.codecogs.com/svg.image?&space;CategoricalCrossEntropy(\vec{y},\vec{\hat{y}})=-\frac{1}{K}\sum_{c=1}^{C}\sum_{k=1}^{K}{y_c_k\log_{e}\hat{y}_i})
