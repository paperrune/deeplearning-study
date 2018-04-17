# CNN-cuda
cuda implementation of [Convolutional Neural Networks](https://github.com/paperrune/Neural-Networks/tree/master/Convolutional_Neural_Networks)</br></br>

## Features
- Multi-GPU is not supported.
- To classify MNIST handwritten digits, following files are required from http://yann.lecun.com/exdb/mnist/
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte
  
- To classify CIFAR-10 datasets, following files are required from https://www.cs.toronto.edu/~kriz/cifar.html
  - data_batch_1.bin
  - data_batch_2.bin
  - data_batch_3.bin
  - data_batch_4.bin
  - data_batch_5.bin
  - test_batch.bin

- The network structure is determined by following variables in the main.cpp.

  ```C++
  string type_layer[] = {"CIFAR-10", "Cbn,ks3", "Cbn,ks3,dw", "Cbn", "Pmax",
				"Cbn,ks3,dw", "Cbn", "Cbn,ks3,dw", "Cbn", "Pmax",
				"Cbn", "Cbn", "Lce,sm"};
  int map_width[]   = { 32, 32, 32, 32, 16,		16, 16, 16, 16,  8,		  1,   1,  1 };
  int map_height[]  = { 32, 32, 32, 32, 16,		16, 16, 16, 16,  8,		  1,   1,  1 };
  int map_depth[]   = {  1,  1,  1,  1,  1,		 1,  1,  1,  1,  1,		  1,   1,  1 };
  int number_maps[] = {  3, 48, 48, 48, 48,		48, 96, 96, 96, 96,		384, 384, 10 };
  ```  
  - There is no type for input layer. "CIFAR-10" is a comment.
  - Type start with 'C(connection/convolution)' and 'P(pooling)' is for hidden layer.
  
  	```
    C(connection/convolution)
    > Activation Function
    "ls"   : Logistic Sigmoid
    "ht"   : Hyperbolic Tangent
    ""     : default is ReLU
    
    > Property
    "dw"      : depthwise separable convolution
    "ksm,n,o" : set kernel width to m, height to n and depth to o [default kernel size : (|map_width[i - 1] - map_width[i]| + 1)*(|map_height[i - 1] - map_height[i]| + 1)*(|map_depth[i - 1] - map_depth[i]| + 1)]
    "ksm"     = "ksm,m,m"
    "stm,n,o"   : set stride width to m, height to n and depth to o [default stride size : 1*1*1]
    "stm"     = "stm,m,m"

    > Regularization
    "bn"   : Batch Normalization
    "do.f" : Dropout with rate 0.f, each neurons is set to zero with a probability of (1 - 0.f)
    ----------------------------------------------------------------------------------------------------
    P(pooling)
    > Type
    "avg"  : Average Pooling
    "max"  : Max Pooling
    
    > Property
    "ksm,n,o" : set pooling width to m, height to n and depth to o [default pooling size : (max(length_map[i - 1], length_map[i]) / min(length_map[i - 1], length_map[i]))^3]
    "ksm"     = "ksm,m,m"
    "stm,n,o" : set stride width to m, height to n and depth to o [default stride size : (max(length_map[i - 1], length_map[i]) / min(length_map[i - 1], length_map[i]))^3]
    "stm"     = "stm,m,m" 
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

## CIFAR-10 classification result
GeForce GTX 1060 6GB</br>
![result](/result.png)
