## CIFAR-10 classification using small VGG-like Convolutional Neural Networks
Following files are required from https://www.cs.toronto.edu/~kriz/cifar.html
  - data_batch_1.bin
  - data_batch_2.bin
  - data_batch_3.bin
  - data_batch_4.bin
  - data_batch_5.bin
  - test_batch.bin

### Compile
- Visual Studio 2015 + CUDA 9.1</br>
![VS_2015](/CIFAR-10_CNN/screenshot/VS_2015.png)</br>
- Python</br>
[Neural_Networks.pyd](https://github.com/paperrune/Neural-Networks/tree/master/Python)</br>

### Results
GeForce GTX 1060 6GB</br>
without data augmentation</br>
C++</br>
![result](/CIFAR-10_CNN/screenshot/CIFAR-10_CNN.png)</br>
Python</br>
![result](/CIFAR-10_CNN/screenshot/CIFAR-10_CNN_Python.png)</br></br>

with data augmentation</br>
![result](/CIFAR-10_CNN/screenshot/CIFAR-10_CNN+.png)
