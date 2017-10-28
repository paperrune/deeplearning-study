# Convolutional Neural Networks
- cuda version: https://github.com/paperrune/CNN-cuda
- Support Batch Normalization and Dropout.
- To classify MNIST handwritten digits, following files are required from http://yann.lecun.com/exdb/mnist/
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte

- The network structure is determined by three variables in the main.cpp.

  ```C++
  88: char *type_layer[]  = {"MNIST", "Cbn", "Pmax", "Cbn", "Pmax", "Cbn", "Lce,sm"};
  91: int map_width[]     = {28, 24, 12,  8,  4,   1,  1};
  92: int map_height[]    = {28, 24, 12,  8,  4,   1,  1};
  93: int number_maps[]   = { 1, 24, 24, 48, 48, 192, 10};
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
    "ksm,n"  : set kernel width to m and height to n  [default kernel size : (map_width[i - 1] - map_width[i] + 1)*(map_height[i - 1] - map_height[i] + 1)]
    "ksm     : same with ksm,m
    "stm,n"  : set stride width to m and height to n  [default stride size : 1*1]
    "stm     : same with stm,m

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
