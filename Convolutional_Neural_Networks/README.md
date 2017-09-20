# Convolutional Neural Networks
- Support Batch Normalization and Dropout.
- To classify MNIST handwritten digits, following files are required from http://yann.lecun.com/exdb/mnist/
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte

- The network structure is determined by three variables in the main.cpp.

  ```C++
  87: char *type_layer[]  = {"MNIST", "Cbn", "Pmax", "Cbn", "Pmax", "Cbn", "Lce,sm"};
  90: int length_map[]    = {28, 24, 12,  8,  4,   1,  1};
  91: int number_map[]    = { 1, 24, 24, 48, 48, 192, 10};
  ```  
  - There is no type for input layer. "MNIST" is a comments.
  - Type start with 'C(connecting/convolution)' and 'P(padding/pooling)' is for hidden layer.  
  
  	```
    C(connecting/convolution)
    > Activation Function
    "ls"   : Logistic Sigmoid
    "ht"   : Hyperbolic Tangent
    ""     : default is ReLU
    
    > Property
    "fsn"  : setting filter size to n^2  [default filter size : (length_map[i - 1] - length_map[i] + 1)^2]
    "stn"  : setting stride to n         [default stride      : 1]

    > Regularization
    "bn"   : Batch Normalization
    "do.f" : Dropout with rate 0.f, each neurons is set to zero with a probability of (1 - 0.f)
    ----------------------------------------------------------------------------------------------------
    P(padding/pooling)
    > Type
    "avg"  : Average Pooling
    "max"  : Max Pooling
    "pad"  : Zero Padding (it should be used to increase the size of the feature map)
    
    stride and pooling size is (length_map[i - 1] / length_map[i])^2 and overlapped pooling is not supported.
	  ```
   - Type start with 'L(loss)' is for output layer.
   
	 ```
	 > Loss Function
	 "ce"  : Cross Entropy
	 "mse" : Mean Squared Error
	 
	 > Activation Function for "ce"
	 "sm"  : Softmax
	 ""    : default is Logistic Sigmoid

	 > Activation Function for "mse"
	 "ht"  : Hyperbolic Tangent
	 "ia"  : Identity Activation f(x) = x
	 ""    : default is Logistic Sigmoid
	 ```
</br>

## MNIST classification result
![result](/Convolutional_Neural_Networks/result.PNG)
