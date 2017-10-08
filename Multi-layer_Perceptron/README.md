# Multi-layer Perceptron

- Support Batch Normalization and Dropout.
- To classify MNIST handwritten digits, following files are required from http://yann.lecun.com/exdb/mnist/
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte

- The network structure is determined by two variables in the main.cpp.

  ```C++
  86: char *type_layer[]   = {"MNIST", "Cbn", "Cbn,do.5", "Lce,sm"};
  91: int number_neurons[] = {28 * 28, 400, 400, 10};
  ```  
  - There is no type for input layer. "MNIST" is a comments.
  - Type start with 'C(connecting)' is for hidden layer.
  
  	```
	> Activation Function
	"ls"   : Logistic Sigmoid
	"ht"   : Hyperbolic Tangent
	""     : default is ReLU
	
	> Regularization
	"bn"   : Batch Normalization
	"do.f" : Dropout with rate 0.f, each neurons is set to zero with a probability of (1 - 0.f)
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

## MNIST classification results
![result](/Multi-layer_Perceptron/result.PNG)
