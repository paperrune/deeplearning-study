import numpy as np
import time
from Neural_Networks import NNGPU

def Read_CIFAR_10(path, number_training, number_test, image, target_output):
    file = open(path + "data_batch_1.bin", 'rb')
    filename = ["data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"]
    index = 0
    
    for h in range(number_training):
        if h % 10000 == 0:
            if not file.closed:
                file.close()
                
            file = open(path + filename[index], 'rb')
            index += 1

        label = int.from_bytes(file.read(1), byteorder='little')

        for j in range(10):
            target_output[h][j] = (j == label)

        for j in range(3 * 32 * 32):
            image[h][j] = int.from_bytes(file.read(1), byteorder='little') / 255

    file.close()
    file = open(path + filename[5], 'rb')

    for h in range(number_training, number_training + number_test):
        label = int.from_bytes(file.read(1), byteorder='little')

        for j in range(10):
            target_output[h][j] = (j == label)

        for j in range(3 * 32 * 32):
            image[h][j] = int.from_bytes(file.read(1), byteorder='little') / 255
            
    file.close()
    

batch_size      = 100
decay_rate      = 0.977
epsilon         = 0.001
learning_rate   = 0.001
number_iterations = 50
number_training = 50000
number_test     = 10000

# train from scratch
NN = NNGPU.Neural_Networks()

NN.Add_Layer("CIFAR-10", 3, 32, 32)
NN.Add_Layer("BN,ReLU", 48, 32, 32) # batch normalization, ReLU activation
NN.Add_Layer("BN,ReLU", 48, 32, 32) # batch normalization, ReLU activation
NN.Add_Layer("BN,ReLU", 48, 32, 32) # batch normalization, ReLU activation
NN.Add_Layer("", 48, 16, 16)
NN.Add_Layer("BN,ReLU", 48, 16, 16) # batch normalization, ReLU activation
NN.Add_Layer("BN,ReLU", 96, 16, 16) # batch normalization, ReLU activation
NN.Add_Layer("BN,ReLU", 96, 16, 16) # batch normalization, ReLU activation
NN.Add_Layer("BN,ReLU", 96, 16, 16) # batch normalization, ReLU activation
NN.Add_Layer("", 96, 8, 8)
NN.Add_Layer("BN,ReLU", 384)        # batch normalization, ReLU activation
NN.Add_Layer("CE,softmax", 10)      # cross-entropy loss, softmax activation

NN.Connect(1, 0, "W,kernel(3x3)")       # 3x3 convolution
NN.Connect(2, 1, "W,kernel(3x3),DS")    # 3x3 depthwise separable convolution
NN.Connect(3, 2, "W")                   # 1x1 convolution
NN.Connect(4, 3, "P,max")               # 2x2 max pooling
NN.Connect(5, 4, "W,kernel(3x3),DS")    # 3x3 depthwise separable convolution
NN.Connect(6, 5, "W")                   # 1x1 convolution
NN.Connect(7, 6, "W,kernel(3x3),DS")    # 3x3 depthwise separable convolution
NN.Connect(8, 7, "W")                   # 1x1 convolution
NN.Connect(9, 8, "P,max")               # 2x2 max pooling
NN.Connect(10, 9, "W")                  # fully connected
NN.Connect(11, 10, "W")                 # fully connected

NN.Initialize(0, 0.01)


# or load pretrained model
# NN = Neural_Networks("CIFAR-10_CNN.txt")

image = np.zeros((number_training + number_test, 3 * 32 * 32), dtype='f')
target_output = np.zeros((number_training + number_test, 10), dtype='f')

path = input("path where CIFAR-10 dataset is : ")
Read_CIFAR_10(path, number_training, number_test, image, target_output)

start = time.time()

for f in range(number_iterations):
    score = [0, 0]

    loss = NN.Train(batch_size, number_training, image, target_output, learning_rate, epsilon)

    h = 0

    for i in range(number_training + number_test):
        h = h + 1
        
        if h == batch_size or i == number_training + number_test - 1:
            _input = np.array(image[i - h + 1])
            output = np.empty((h, 10), dtype='f')

            for g in range(1, h):
                _input = np.append(_input, image[i - h + g + 1])

            NN.Test(h, _input, output)
            
            for g in range(h):                
                score[0 if i - h + g + 1 < number_training else 1] += int(target_output[i - h + g + 1][np.argmax(output[g])])
                
            h = 0

    print('.', end='')
    NN.Save("NN.txt")
    print('score: {} / {}, {} / {}, loss = {:.6f}, step {}, {:.2f} sec'.format(score[0], number_training, score[1], number_test, loss, f + 1, time.time() - start))
    
    learning_rate *= decay_rate

