import numpy as np
import time
from Neural_Networks import NNCPU

def Read_MNIST(training_set_images, training_set_labels, test_set_images, test_set_labels, number_training, number_test, image, target_output):
    # training set images
    file = open(training_set_images, 'rb')

    for _ in range(4):
        file.read(4)
        
    for h in range(number_training):
        for j in range(28 * 28):
            image[h][j] = int.from_bytes(file.read(1), byteorder='little') / 255
            
    file.close()

    # training set labels
    file = open(training_set_labels, 'rb')

    for _ in range(2):
        file.read(4)
        
    for h in range(number_training):
        label = int.from_bytes(file.read(1), byteorder='little')
        
        for j in range(10):
            target_output[h][j] = float(j == label)
            
    file.close()

    # test set images
    file = open(test_set_images, 'rb')

    for _ in range(4):
        file.read(4)
        
    for h in range(number_training, number_training + number_test):        
        for j in range(28 * 28):
            image[h][j] = int.from_bytes(file.read(1), byteorder='little') / 255
            
    file.close()

    # test set labels
    file = open(test_set_labels, 'rb')

    for _ in range(2):
        file.read(4)
        
    for h in range(number_training, number_training + number_test):
        label = int.from_bytes(file.read(1), byteorder='little')
        
        for j in range(10):
            target_output[h][j] = float(j == label)
            
    file.close()
    

batch_size      = 60
decay_rate      = 0.977
epsilon         = 0.001
learning_rate   = 0.005
noise           = 0.001
number_iterations = 50
number_output   = 10
number_training = 60000
number_test     = 10000
time_step       = 28

# train from scratch
NN = NNCPU.Neural_Networks(time_step)

NN.Add_Layer("MNIST", 784 // time_step)
NN.Add_Layer("BN,LSTM", 100)            # batch normalization, LSTM forward direction
NN.Add_Layer("BN,LSTM,backward", 100)   # batch normalization, LSTM backward direction
NN.Add_Layer("CE,softmax", 10)          # cross-entropy loss, softmax activation

NN.Connect(1, 0, "W")           # fully connected
NN.Connect(1, 1, "W,recurrent") # fully connected
NN.Connect(2, 0, "W")           # fully connected
NN.Connect(2, 2, "W,recurrent") # fully connected
NN.Connect(3, 1, "W")           # fully connected
NN.Connect(3, 2, "W")           # fully connected

NN.Initialize(0, 0.1, 0.1)

time_mask = np.empty(time_step, dtype='b')

for t in range(time_step):
    time_mask[t] = (t == 0 or t == time_step - 1)
    
NN.Set_Time_Mask(3, time_mask)


# or load pretrained model
# NN = Neural_Networks("MNIST_BLSTM.txt")

NN.Set_Number_Threads(int(input("The number of threads : ")))

image = np.zeros((number_training + number_test, 784), dtype='f')
target_output = np.zeros((number_training + number_test, time_step * 10), dtype='f')

path = input("path where MNIST handwritten digits dataset is : ")
Read_MNIST(path + "train-images.idx3-ubyte", path + "train-labels.idx1-ubyte", path + "t10k-images.idx3-ubyte", path + "t10k-labels.idx1-ubyte", number_training, number_test, image, target_output)

for h in range(number_training + number_test):
    index = (time_step - 1) * number_output
    
    for j in range(number_output):
        target_output[h][index + j] = target_output[h][j]

start = time.time()

for f in range(number_iterations):
    score = [0, 0]

    loss = NN.Train(batch_size, number_training, image, target_output, learning_rate, epsilon, noise)

    h = 0

    for i in range(number_training + number_test):
        h = h + 1
        
        if h == batch_size or i == number_training + number_test - 1:
            _input = np.array(image[i - h + 1])
            output = np.empty((h, time_step * number_output), dtype='f')

            for g in range(1, h):
                _input = np.append(_input, image[i - h + g + 1])

            NN.Test(h, _input, output)
            
            for g in range(h):
                index = (time_step - 1) * number_output
                output_mean = np.empty(number_output, dtype='f')
            
                for j in range(number_output):
                    output_mean[j] = (output[g][j] + output[g][index + j]) / 2
            
                score[0 if i - h + g + 1 < number_training else 1] += int(target_output[i - h + g + 1][np.argmax(output_mean)])
                
            h = 0

    print('.', end='')
    NN.Save("NN.txt")
    print('score: {} / {}, {} / {}, loss = {:.6f}, step {}, {:.2f} sec'.format(score[0], number_training, score[1], number_test, loss, f + 1, time.time() - start))
    
    learning_rate *= decay_rate
