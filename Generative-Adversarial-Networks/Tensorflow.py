import numpy as np
import tensorflow as tf

from PIL import Image

batch_size = 128
input_dim = 784
iteration = 100000
learning_rate = 0.0001
noise_dim = 100
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train = x_train.reshape([x_train.shape[0], -1]).astype('float32') / 255 
 
# input placeholders
X = tf.placeholder(tf.float32, [None, input_dim])
Z = tf.placeholder(tf.float32, [None, noise_dim])

# variable & model
with tf.variable_scope('generator'):
    W1 = tf.get_variable(name="W1", shape=[noise_dim, 512], initializer=tf.keras.initializers.he_normal())
    b1 = tf.get_variable(name="b1", shape=[512], initializer=tf.zeros_initializer())
    W2 = tf.get_variable(name="W2", shape=[512, input_dim], initializer=tf.glorot_uniform_initializer())
    b2 = tf.get_variable(name="b2", shape=[input_dim], initializer=tf.zeros_initializer())

with tf.variable_scope('discriminator'):
    W3 = tf.Variable(tf.random_uniform([input_dim, 512], minval=-0.01, maxval=0.01))
    b3 = tf.Variable(tf.zeros([512]))
    W4 = tf.Variable(tf.random_uniform([512, 1], minval=-0.01, maxval=0.01))
    b4 = tf.Variable(tf.zeros([1]))
 
def generator(Z):
    L1 = tf.nn.relu(tf.matmul(Z, W1) + b1)
    L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
    
    return L2

def discriminator(X):
    L1 = tf.nn.relu(tf.matmul(X, W3) + b3)
    L2 = tf.matmul(L1, W4) + b4
    
    return L2, tf.nn.sigmoid(L2)

G = generator(Z)
D, D_output = discriminator(X)
DG, DG_output = discriminator(G)

# accuracy & loss & optimizer
d_fake_acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(DG_output), tf.zeros_like(DG_output)), tf.float32))
d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DG, labels=tf.zeros_like(DG)))

d_real_acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(D_output), tf.ones_like(D_output)), tf.float32))
d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D, labels=tf.ones_like(D)))

g_acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(DG_output), tf.ones_like(DG_output)), tf.float32))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DG, labels=tf.ones_like(DG)))

t_var = tf.trainable_variables()
d_var = [var for var in t_var if 'discriminator' in var.name]
g_var = [var for var in t_var if 'generator' in var.name]

d_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(0.5 * d_fake_loss + 0.5 * d_real_loss, var_list=d_var)
g_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_var)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
     
    for i in range(iteration):
        index = np.random.randint(0, len(x_train) - batch_size + 1)
        loss  = [None] * 3
        noise = np.random.uniform(-1, 1, size=(batch_size, noise_dim))
        score = [None] * 3

        x_batch = x_train[index:index + batch_size]
        loss[0], loss[1], score[0], score[1], _ = sess.run([d_fake_loss, d_real_loss, d_fake_acc, d_real_acc, d_train], feed_dict={X: x_batch, Z: noise})
        loss[2], score[2], _ = sess.run([g_loss, g_acc, g_train], feed_dict={Z: noise})

        print(i + 1, 'D_fake:', [loss[0], score[0]], '\tD_real:', [loss[1], score[1]], '\tG:', [loss[2], score[2]])

        if i % 10000 == 9999:
            array = sess.run(G, feed_dict={Z: noise})
            batch = []

            for h in range(0, batch_size, 16):
                if h + 16 <= batch_size:
                    batch.append(array[h:h + 16])

            array = np.array(batch)
            array = array.reshape((array.shape[0], array.shape[1], 28, 28))
            array = np.transpose(array, (0, 2, 1, 3))
            array = array.reshape((array.shape[0] * 28, array.shape[2] * 28))
            array = np.uint8(array * 255)
            
            image = Image.fromarray(array)
            image.save('{}.png'.format(i + 1))
