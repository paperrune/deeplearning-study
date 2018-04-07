# Neural-Networks
C++ implementation of Neural Networks</br></br>

## Property
### Layer
```C++
> Activation Function
"ELUf"    : exponential linear unit with hyperparameter f
"PReLU.f" : parametric ReLU with initial negative slope 0.f
"ReLU"    : rectified linear unit
"ReLU.f"  : leaky ReLU with negative slope 0.f
"sigmoid" : logistic sigmoid
"softmax" : softmax
"tangent" : hyperbolic tangent

> Loss
"CE"  : cross-entopy
"MSE" : mean squared error

> Regularization
"BN"        : batch normalization
"dropout.f" : dropout with rate 0.f, each neuron is set to zero with a probability of (1 - 0.f)
```

### Connection
```C++
"W" : connect with weights
