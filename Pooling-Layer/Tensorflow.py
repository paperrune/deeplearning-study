import tensorflow as tf
import time
 
batch_size = 128
epochs = 30
learning_rate = 0.05
momentum = 0.9
 
# input image dimensions
img_rows, img_cols = 28, 28
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 
x_train = x_train.reshape([x_train.shape[0], img_rows, img_cols, 1]).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
 
x_test = x_test.reshape([x_test.shape[0], img_rows, img_cols, 1]).astype('float32') / 255
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
 
# input place holders
X = tf.placeholder(tf.float32, [None, img_rows, img_cols, 1])
Y = tf.placeholder(tf.float32, [None, 10])
 
# weights & bias for networks
W1 = tf.Variable(tf.random_uniform([5, 5, 1, 24], minval=-0.1, maxval=0.1))
b1 = tf.Variable(tf.zeros([24]))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID')
L1 = tf.nn.relu(tf.nn.bias_add(L1, b1))
P1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
W2 = tf.Variable(tf.random_uniform([5, 5, 24, 48], minval=-0.1, maxval=0.1))
b2 = tf.Variable(tf.zeros([48]))
L2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='VALID')
L2 = tf.nn.relu(tf.nn.bias_add(L2, b2))
P2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
P2_flat = tf.reshape(P2, [-1, 4 * 4 * 48])
 
W3 = tf.Variable(tf.random_uniform([4 * 4 * 48, 512], minval=-0.01, maxval=0.01))
b3 = tf.Variable(tf.zeros([512]))
L3 = tf.nn.relu(tf.matmul(P2_flat, W3) + b3)
 
W4 = tf.Variable(tf.random_uniform([512, 10], minval=-0.01, maxval=0.01))
b4 = tf.Variable(tf.zeros([10]))
 
hypothesis = tf.matmul(L3, W4) + b4
 
# define cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost)
 
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
 
    for step in range(epochs):
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
