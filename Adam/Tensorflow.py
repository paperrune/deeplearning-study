import numpy as np
import tensorflow as tf
import time
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 
x_train = x_train.reshape([x_train.shape[0], -1]).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
 
x_test = x_test.reshape([x_test.shape[0], -1]).astype('float32') / 255
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
 
batch_size = 128
epochs = 30
learning_rate = 0.0005
 
# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
 
# weights & bias for networks
W1 = tf.Variable(tf.random_uniform([784, 512], minval=-0.01, maxval=0.01))
b1 = tf.Variable(tf.zeros([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
 
W2 = tf.Variable(tf.random_uniform([512, 512], minval=-0.01, maxval=0.01))
b2 = tf.Variable(tf.zeros([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
 
W3 = tf.Variable(tf.random_uniform([512, 10], minval=-0.01, maxval=0.01))
b3 = tf.Variable(tf.zeros([10]))
 
hypothesis = tf.matmul(L2, W3) + b3
 
# cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)
 
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
 
    for step in range(epochs):
        # shuffle data
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]
        
        loss = [0, 0]
        score = [0, 0]
 
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            
            c, a, _ = sess.run([cost, accuracy, train], feed_dict={X: x_batch, Y: y_batch})
            loss[0] += c * len(x_batch)
            score[0] += a * len(x_batch)
 
        for i in range(0, x_test.shape[0], batch_size):
            x_batch = x_test[i:i + batch_size]
            y_batch = y_test[i:i + batch_size]
            
            c, a = sess.run([cost, accuracy], feed_dict={X: x_batch, Y: y_batch})
            loss[1] += c * len(x_batch)
            score[1] += a * len(x_batch)
            
        print('loss: {:.4f} / {:.4f}\taccuracy: {:.4f} / {:.4f}\tstep {}  {:.2f} sec'.format(loss[0] / x_train.shape[0], loss[1] / x_test.shape[0], score[0] / x_train.shape[0], score[1] / x_test.shape[0], step + 1, time.time() - start_time))
