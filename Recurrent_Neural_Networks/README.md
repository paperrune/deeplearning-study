# Recurrent Neural Networks
- Support Batch Normalization and Dropout.
- To classify MNIST handwritten digits, following files are required from http://yann.lecun.com/exdb/mnist/
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte

- The network structure is determined by four variables in the main.cpp.

  ```C++
  91: char *type_layer[]  = {"MNIST", "Clstm", "Lce,sm"};
  94: int length_train    = 28;
  96: int length_map[]    = {1, 1, 1};
  97: int number_maps[]   = {784 / length_train, 100, 10};
  ```  
  - The type is extended in [Convolutional Neural Networks](https://github.com/paperrune/Neural-Networks/tree/master/Convolutional_Neural_Networks), and only new types are shown on this page.
 
  	```
    C(connecting/convolution)
    > Recurrent
    "rc"   : vanilla recurrent
    "gru"  : gated recurrent unit
    "lstm" : long-short term memory
   
    > Regularization
    "bn"   : Batch Normalization
             (https://arxiv.org/abs/1603.09025.pdf)
    "do.f" : Dropout with rate 0.f, each neurons is set to zero with a probability of (1 - 0.f)
             (https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
	  ``` 
</br>

## MNIST classification results
![result](/Recurrent_Neural_Networks/result.PNG)
