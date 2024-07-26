# List of error calculus functions and their derivatives. 
These functions give the error curve to be applied to the backpropagation algorithm. To search for a local or global minimum with luck. 

## Mean Square Error method for regression

 $$MSE(\vec{y},\vec{\hat{y}})=\frac{1}{K}\sum_{k=1}^{K}( y_i - \hat{y}_i)^{2}$$
 
Old ANN use it even for classification but it is not the best for this purpose.
This method is based on the gausian distribution (bell curve).
MSE is the value for the entire output layer, it is noy the individual error for each neuron.

## Binary Cross Entropy method for Classification

![Bin](https://latex.codecogs.com/svg.image?&space;BinaryCrossEntropy(\vec{y},\vec{\hat{y}})=\frac{1}{K}\sum_{k=1}^{K}{y_i\log_{e}\hat{y}_i&plus;(1-y_i)\log_{e}(1-y_i)})

## Categorical Cross Entropy method for Classification must end only in one category 

![Cat](https://latex.codecogs.com/svg.image?&space;CategoricalCrossEntropy(\vec{y},\vec{\hat{y}})=-\frac{1}{K}\sum_{c=1}^{C}\sum_{k=1}^{K}{y_c_k\log_{e}\hat{y}_i})
