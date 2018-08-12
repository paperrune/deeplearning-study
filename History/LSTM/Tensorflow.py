import numpy as np
import tensorflow as tf
import time
 
batch_size = 128
decay = 1e-6
epochs = 30
initial_learning_rate = 0.1
momentum = 0.9

learning_rate = tf.placeholder(tf.float32, shape=[])
 
# input image dimensions
number_input = 28
time_step = 28
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 
x_train = x_train.reshape([x_train.shape[0], time_step, number_input])
x_train = x_train.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
 
x_test = x_test.reshape([x_test.shape[0], time_step, number_input])
x_test = x_test.astype('float32') / 255
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
 
# input place holders
X = tf.placeholder(tf.float32, [None, time_step, number_input])
Y = tf.placeholder(tf.float32, [None, 10])
 
# construct neural networks
cell_fw = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1)
cell_bw = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1)
(output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X, dtype=tf.float32)

output_fw = tf.transpose(output_fw, [1, 0, 2])
output_bw = tf.transpose(output_bw, [1, 0, 2])
outputs = tf.concat([output_fw[-1], output_bw[0]], axis=1)

W = tf.get_variable(name="W", shape=[256, 10], initializer=tf.glorot_uniform_initializer())
b = tf.get_variable(name="b", shape=[10], initializer=tf.zeros_initializer())
 
hypothesis = tf.matmul(outputs, W) + b
 
# define cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost)
 
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
 
    iterations = 0
 
    for step in range(epochs):
        # shuffle data
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]
        
        score = [0, 0]
        loss = [0, 0]
 
        for i in range(0, x_train.shape[0], batch_size):
            size = batch_size if i + batch_size <= x_train.shape[0] else x_train.shape[0] - i
            
            c, a, _ = sess.run([cost, accuracy, train], feed_dict={X: x_train[i:i+size], Y: y_train[i:i+size], learning_rate: initial_learning_rate / (1 + decay * iterations)})
            loss[0] += c * size
            score[0] += a * size
            iterations += 1
 
        for i in range(0, x_test.shape[0], batch_size):
            size = batch_size if i + batch_size <= x_test.shape[0] else x_test.shape[0] - i
            
            c, a = sess.run([cost, accuracy], feed_dict={X: x_test[i:i+size], Y: y_test[i:i+size]})
            loss[1] += c * size
            score[1] += a * size
            
        print('loss: {:.4f} / {:.4f}\taccuracy: {:.4f} / {:.4f}\tstep {}  {:.2f} sec'.format(loss[0] / x_train.shape[0], loss[1] / x_test.shape[0], score[0] / x_train.shape[0], score[1] / x_test.shape[0], step + 1, time.time() - start_time))
