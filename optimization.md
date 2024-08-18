# Optimization with backpropagation

**Backpropagation** is a gradient estimation method used to train neural network models.
The gradient estimate is used by the **optimization algorithm** to compute the network parameter updates.
(optimization = look for the maximum or minimum point in math).

Backpropagation calculates the gradient of a loss function with respect to the weights of the network for a single sample input–output (training set),
The equations are for one single layer output layer or hidden layer,
And The magic happens moving backward from the output layer to the first layer.
Moving from the output layer to the first layer let us avoid redundant calculations of intermediate terms in the chain rule;

Backpropagation is backward propagation of errors.

## Gradient descent

Optimizing an objective function.
In the practic  SGD or Adam. **Do not use Adam by default. Check the performance**

### Stochastic gradient descent

SGD is an algorith for optimizing an objective function.
It can be regarded as a stochastic approximation of gradient descent optimization, since it replaces the actual gradient (calculated from the entire data set)
by an estimate thereof (calculated from a randomly selected subset of the data).


For any layer **l**, with **j** entries, with weights **W**, in the epoch **t**, for the neuron **k** in the layer, using the loss function **L**.
Using the learning rate **eta**.
In the epoch t=0, the values of W are defined randomly.

![sdasd](https://latex.codecogs.com/svg.image?W^{(l)}_{jkt}=W^{(l)}_{jk(t-1)}-\eta\frac{\partial&space;L}{\partial&space;W^{(l)}_{jk(t-1)}})

For the bias

![sdasd](https://latex.codecogs.com/svg.image?b^{(l)}_{kt}=b^{(l)}_{k(t-1)}-\eta\frac{\partial&space;L}{\partial&space;b^{(l)}_{k(t-1)}})

Explore **eta** values around 0.01.

### Learning rate scheduling
When **eta** is too big the optimisation diverges and when it is too little you need a whole life to approach the solution.
This is when the learning rate is a function depending on the epoch.

#### Exponential (the belly hit algorithm !!)

$$ \eta(t)=A\cdot e^{-kt}$$ 

In the above equation we see a very aggressive curve,
like when you jump from a diving board into a swimming pool and you hit your belly.

#### Step-based
if you want to smooth the curve you use it:

$$ \eta(t)= \frac{A}{k\cdot t+1}$$

#### Error-based

The higher the error, the higher the learning rate.
In the peactice people use AdaGrad (for adaptive gradient algorithm).
The Epsilon value avoid the division by zero.
Some times:
$$\epsilon=10^{-8}$$
Some times:
$$\epsilon=10^{-10}$$


$$cache_t=cache_{t-t}+(\frac{\partial L}{\partial W^{(l)}_{jk(t-1)}})^2$$

![sdasd](https://latex.codecogs.com/svg.image?W^{(l)}_{jkt}=W^{(l)}_{jk(t-1)}-\eta\frac{\partial&space;L}{\partial&space;W^{(l)}_{jk(t-1)}}\cdot&space;(\frac{1}{\sqrt{cache&plus;\epsilon}}))

Geoff Hinton said that the reason why adagrad works is that it decreases the learning rate is that the cache grows very fast,
as fast as the strength of the gradient. 
Geoff Hinton did not like the method and invented the RMSProp.

#### Error-based (RMSProp)

$$decay=0.99$$ 

or

$$decay=0.999$$

$$cache_t=cache_{t-t} \cdot decay + (1-decay) \cdot (\frac{\partial L}{\partial W^{(l)}_{jk(t-1)}})^2$$

### ADAM Adaptative moment estimation

The idea of this algorithm is to filter out those points where the error is unfairly high,
in other words it wants to remove abrupt changes, high frequencies.
It also allows slight changes in error to be taken into account, i.e. to let low frequencies through.

A good reader will think: "But what the fuck are you talking about?
What fucking memory can hold all that information to be able to tell if I'm having errors at a high or low frequency?"

It is not possible to calculate things with all the epoch history because the memory is not enough.
But, these smart-asses know probability and statistics, and use equations to calculate the nth moment of a random variable.

The nth moment of a random variable comes from the equations of the second momen:
The variance is the second movement minus the mean squared.

**constants**

$$\beta_1=0.9$$

$$\beta_2=0.999$$

$$\epsilon=10^{-7}$$

$$\epsilon=10^{-8}$$

$$\eta=10^{-3}$$

The **first moment** (or momentum) of the gradient of the loss function respect the weights is $$m_t$$

$$m_0 = 0 $$

$$m_t=(\beta_1 \cdot m_{t-1}) + (1-\beta_1) \cdot \frac{\partial L}{\partial W^{(l)}_{jk(t-1)}}$$

The **second moment** (or velocity) of the gradient of the loss function respect the weights is $$v_t$$

$$v_0 = 0 $$

$$v_t=(\beta_2 \cdot v_{t-1}) + (1-\beta_2) \cdot (\frac{\partial L}{\partial W^{(l)}_{jk(t-1)}})^2$$

Next **compute bias-corrected first moment**
In the next step, the value of the power of beta is equal to two and is not t (the epoch) when it is put into a loop until
the new values of the weights converge. I'm not going to do that variant of the algorithm, because I think it sucks.

![sdasd](https://latex.codecogs.com/svg.image?\hat{m}_t=\frac{m_{t-1}}{1-\beta_1^{t}})


Next **compute bias-corrected second moment**
In the next step, the value of the power of beta is equal to two and is not t (the epoch) when it is put into a loop until
the new values of the weights converge. I'm not going to do that variant of the algorithm, because I think it sucks.

![sdasd](https://latex.codecogs.com/svg.image?\hat{v}_t=\frac{v_{t-1}}{1-\beta_2^{t}})

And then integrate the values to optimization

![sdasd](https://latex.codecogs.com/svg.image?W^{(l)}_{jkt}=W^{(l)}_{jk(t-1)}-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}&plus;\epsilon})
