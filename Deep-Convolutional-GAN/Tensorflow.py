import numpy as np
import tensorflow as tf

from PIL import Image

batch_size = 128
decay = 0.99
input_dim = (32, 32, 3)
iteration = 100000
learning_rate = 0.0001
noise_dim = 100

is_training = tf.placeholder(tf.bool)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 
x_train = x_train.reshape([x_train.shape[0]] + list(input_dim)).astype('float32') / 255
x_train = 2 * x_train - 1
 
# input placeholders
X = tf.placeholder(tf.float32, [None] + list(input_dim))
Z = tf.placeholder(tf.float32, [None, noise_dim])

# variable & model
with tf.variable_scope('generator'):
    W1 = tf.get_variable(name="W1", shape=[noise_dim, 4 * 4 * 256], initializer=tf.keras.initializers.he_normal())
    b1 = tf.get_variable(name="b1", shape=[4 * 4 * 256], initializer=tf.zeros_initializer())

    W2 = tf.get_variable(name="W2", shape=[5, 5, 128, 256], initializer=tf.keras.initializers.he_normal())
    b2 = tf.get_variable(name="b2", shape=[128], initializer=tf.zeros_initializer())

    W3 = tf.get_variable(name="W3", shape=[5, 5, 64, 128], initializer=tf.keras.initializers.he_normal())
    b3 = tf.get_variable(name="b3", shape=[64], initializer=tf.zeros_initializer())

    W4 = tf.get_variable(name="W4", shape=[5, 5, 3, 64], initializer=tf.glorot_uniform_initializer())
    b4 = tf.get_variable(name="b4", shape=[3], initializer=tf.zeros_initializer())

with tf.variable_scope('discriminator'):
    W5 = tf.Variable(tf.random_uniform([5, 5, 3, 64], minval=-0.01, maxval=0.01))
    b5 = tf.Variable(tf.zeros([64]))

    W6 = tf.Variable(tf.random_uniform([5, 5, 64, 128], minval=-0.01, maxval=0.01))
    b6 = tf.Variable(tf.zeros([128]))

    W7 = tf.Variable(tf.random_uniform([5, 5, 128, 256], minval=-0.01, maxval=0.01))
    b7 = tf.Variable(tf.zeros([256]))

    W8 = tf.Variable(tf.random_uniform([4 * 4 * 256, 1], minval=-0.01, maxval=0.01))
    b8 = tf.Variable(tf.zeros([1]))
 
def generator(Z):
    with tf.variable_scope('generator'):
        L1 = tf.matmul(Z, W1) + b1
        L1 = tf.contrib.layers.batch_norm(L1, center=True, decay=decay, scale=True, is_training=is_training)
        L1 = tf.nn.relu(L1)
        L1 = tf.reshape(L1, [-1, 4, 4, 256])
        # 4, 4, 256 /
        
        L2 = tf.nn.conv2d_transpose(L1, W2, strides=[1, 2, 2, 1], output_shape=[batch_size, 8, 8, 128], padding='SAME')
        L2 = tf.nn.bias_add(L2, b2)
        L2 = tf.contrib.layers.batch_norm(L2, center=True, decay=decay, scale=True, is_training=is_training)
        L2 = tf.nn.relu(L2)
        # 8, 8, 128 /
        
        L3 = tf.nn.conv2d_transpose(L2, W3, strides=[1, 2, 2, 1], output_shape=[batch_size, 16, 16, 64], padding='SAME')
        L3 = tf.nn.bias_add(L3, b3)
        L3 = tf.contrib.layers.batch_norm(L3, center=True, decay=decay, scale=True, is_training=is_training)
        L3 = tf.nn.relu(L3)
        # 16, 16, 64 /

        L4 = tf.nn.conv2d_transpose(L3, W4, strides=[1, 2, 2, 1], output_shape=[batch_size, 32, 32, 3], padding='SAME')
        L4 = tf.nn.bias_add(L4, b4)
        L4 = tf.tanh(L4)
        # 32, 32, 3 /

    return L4

def discriminator(X):    
    with tf.variable_scope('discriminator'):
        L1 = tf.nn.conv2d(X, W5, strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.bias_add(L1, b5)
        L1 = tf.contrib.layers.batch_norm(L1, center=True, decay=decay, reuse=tf.AUTO_REUSE, scale=True, scope='BN1', is_training=is_training)
        L1 = tf.nn.relu(L1)
        # 16, 16, 64 /

        L2 = tf.nn.conv2d(L1, W6, strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.bias_add(L2, b6)
        L2 = tf.contrib.layers.batch_norm(L2, center=True, decay=decay, reuse=tf.AUTO_REUSE, scale=True, scope='BN2', is_training=is_training)
        L2 = tf.nn.relu(L2)
        # 8, 8, 128 /

        L3 = tf.nn.conv2d(L2, W7, strides=[1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.bias_add(L3, b7)
        L3 = tf.contrib.layers.batch_norm(L3, center=True, decay=decay, reuse=tf.AUTO_REUSE, scale=True, scope='BN3', is_training=is_training)
        L3 = tf.nn.relu(L3)
        # 4, 4, 256 /

        L4 = tf.reshape(L3, [-1, 4 * 4 * 256])
        L4 = tf.matmul(L4, W8) + b8
    
    return L4, tf.nn.sigmoid(L4)

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

u_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
d_ops = [var for var in u_ops if 'discriminator' in var.name]
g_ops = [var for var in u_ops if 'generator' in var.name]

with tf.control_dependencies(d_ops):
    d_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(0.5 * d_fake_loss + 0.5 * d_real_loss, var_list=d_var)
with tf.control_dependencies(g_ops):
    g_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_var)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
     
    for i in range(iteration):
        index = (i * batch_size) % (len(x_train) - batch_size)
        loss  = [None] * 3
        noise = np.random.uniform(-1, 1, size=(batch_size, noise_dim))
        score = [None] * 3

        x_batch = x_train[index:index + batch_size]
        loss[0], loss[1], score[0], score[1], _ = sess.run([d_fake_loss, d_real_loss, d_fake_acc, d_real_acc, d_train], feed_dict={X: x_batch, Z: noise, is_training: True})
        loss[2], score[2], _ = sess.run([g_loss, g_acc, g_train], feed_dict={Z: noise, is_training: True})

        print(i, 'D_fake:', [loss[0], score[0]], '\tD_real:', [loss[1], score[1]], '\tG:', [loss[2], score[2]])

        if i % 10000 == 9999:
            array = sess.run(G, feed_dict={Z: noise, is_training: False})
            batch = []

            for h in range(0, batch_size, 16):
                if h + 16 <= batch_size:
                    batch.append(array[h:h + 16])

            array = np.array(batch)
            array = array.reshape((array.shape[0], array.shape[1]) + input_dim)
            array = np.transpose(array, (0, 2, 1, 3, 4))
            array = array.reshape((array.shape[0] * input_dim[0], array.shape[2] * input_dim[1], input_dim[2]))
            array = np.uint8((array + 1) * 255 / 2)
            
            image = Image.fromarray(array)
            image.save('{}.png'.format(i + 1))
