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

## Notation

$$W_{ij}^{(l)}$$ Weight from neuron j in layer l-1 to neuron i in layer l.

$$b_{i}^{(l)}$$ : Bias for neuron i in layer l.

$$x_{i}^{(l)}$$ : Activation of neuron i in layer l.

$$z_{i}^{(l)}$$ : Pre-activation of neuron i in **hidden layer** l, with j entries, where 

![dsda](https://latex.codecogs.com/svg.image?z_i^{(l)}=\sum&space;_j&space;W_{ij}^{(l)}x_j^{(l-1)}&plus;b_i^{(l)})

$$z_{k}^{(L)}$$ : Pre-activation of neuron k in **output layer** L, with j entries, where 

![dsda](https://latex.codecogs.com/svg.image?z_k^{(L)}=\sum&space;_j&space;W_{kj}^{(L)}x_j^{(L-1)}&plus;b_k^{(L)})

$$\sigma_{i}^{(l)}(z_{i}^{(l)})$$: Activation function in **hidden layer** l.

![fsdf](https://latex.codecogs.com/svg.image?\hat{y}_k=\sigma_{k}^{(L)}(z_{k}^{(L)}))​: Activation function in **output layer** L..Predicted output for neuron i.


$$y_k$$: True label for neuron k. The expected value.

$$\text{MSE} = \frac{1}{k} \sum_{i=1}^{k} (y_k - \hat{y}_k)^2$$ : the Mean Square Error (MSE) With n neurons in the last layer.

## Backward Pass: Derivatives
### Derivative of MSE with Respect to Output Layer Weights $$W_{jk}^{(L)}$$

Chain Rule for Partial Derivative of MSE with Respect to $$W_{jk}$$​:

![ahh](https://latex.codecogs.com/svg.image?\frac{\partial\text{MSE}}{\partial&space;W_{jk}}=\frac{\partial\text{MSE}}{\partial\hat{y}_k}\cdot\frac{\partial\hat{y}_k}{\partial&space;z_k}\cdot\frac{\partial&space;z_k}{\partial&space;W_{jk}})

Chain Rule for Partial Derivative of MSE with Respect to $$b_{k}$$​:

![ahh](https://latex.codecogs.com/svg.image?\frac{\partial\text{MSE}}{\partial&space;b_k}=\frac{\partial\text{MSE}}{\partial\hat{y}_k}\cdot\frac{\partial\hat{y}_k}{\partial&space;z_k}\cdot\frac{\partial&space;z_k}{\partial&space;b_k})

Auxiliar calculous:

$$\frac{\partial \hat{y}_i}{\partial z_i^{(L)}} = \sigma'(z_i^{(L)})$$

$$\frac{\partial z_k}{\partial b_k} = 1$$

$$\frac{\partial z_k}{\partial W_{kj}} = x_{j}^{(L-1)}$$

![sdasda](https://latex.codecogs.com/svg.image?\frac{\partial\text{MSE}}{\partial\hat{y}_k^{(L)}}=\frac{1}{K}\frac{\partial}{\partial\hat{y}_k^{(L)}}((y_1-\hat{y}_1)^2&plus;...&plus;(y_k-\hat{y}_k)^2&plus;...&plus;(y_K-\hat{y}_K)^2))

The other part that are not k, are 0. And apply substitution. with the chain rule to get:

$$\frac{\partial \text{MSE}}{\partial \hat{y}_k^{(L)}} = \frac{-2}{n} (y_k - \hat{y}_k )$$

#### Error Signal for Output Neuron k:

![dadasda](https://latex.codecogs.com/svg.image?\delta_k^{(L)}=\frac{\partial\text{MSE}}{\partial\hat{y}_k}\cdot\frac{\partial\hat{y}_k}{\partial&space;z_k}=-\frac{2}{K}(y_k-\hat{y}_k)\cdot\sigma'(z_k^{(L)}))

#### Gradient for Weights $$W_{jk}^{(L)}$$

![adsasd](https://latex.codecogs.com/svg.image?\frac{\partial\text{MSE}}{\partial&space;W_{jk}^{(L)}}=\delta_k^{(L)}\cdot\frac{\partial&space;z_k}{\partial&space;W_{jk}^{(L)}}=\delta_k^{(L)}\cdot&space;x_{j}^{(L-1)})

### Derivative of MSE with Respect to Hideen Layer Weights $$W_{ij}^{(l)}$$


#### Error Signal for hidden Neuron i:


## Binary Cross Entropy method for Classification

![Bin](https://latex.codecogs.com/svg.image?&space;BinaryCrossEntropy(\vec{y},\vec{\hat{y}})=\frac{1}{K}\sum_{k=1}^{K}{y_i\log_{e}\hat{y}_i&plus;(1-y_i)\log_{e}(1-y_i)})

This method is based on the bernoulli distribution .
Binary Cross Entropy is the value for the entire output layer, it is not the individual error for each neuron.

## Categorical Cross Entropy method for Classification must end only in one category 

![Cat](https://latex.codecogs.com/svg.image?&space;CategoricalCrossEntropy(\vec{y},\vec{\hat{y}})=-\frac{1}{K}\sum_{c=1}^{C}\sum_{k=1}^{K}{y_c_k\log_{e}\hat{y}_i})
