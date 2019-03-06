import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
def plot_results(models, data, session):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        session (class): tensorflow.python.client.session.Session
    """
    
    encoder, decoder = models
    x_test, y_test = data
    sess = session
        
    filename = 'vae_mean.png'
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = sess.run(encoder, feed_dict={X: x_test})
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

    filename = 'digits_over_latent.png'
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = sess.run(decoder, feed_dict={Z: z_sample})
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.tight_layout()
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
   

batch_size = 128
epochs = 30
input_dim = 784
latent_dim = 2
learning_rate = 0.001
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 
x_train = x_train.reshape([x_train.shape[0], -1]).astype('float32') / 255 
x_test = x_test.reshape([x_test.shape[0], -1]).astype('float32') / 255
 
# input place holders
X = tf.placeholder(tf.float32, [None, input_dim])
Z = tf.placeholder(tf.float32, [None, latent_dim])
 
# encoder
W1 = tf.get_variable(name="W1", shape=[input_dim, 512], initializer=tf.keras.initializers.he_normal())
b1 = tf.get_variable(name="b1", shape=[512], initializer=tf.zeros_initializer())
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable(name="W2", shape=[512, latent_dim], initializer=tf.glorot_uniform_initializer())
b2 = tf.get_variable(name="b2", shape=[latent_dim], initializer=tf.zeros_initializer())
L2 = tf.matmul(L1, W2) + b2

W3 = tf.get_variable(name="W3", shape=[512, latent_dim], initializer=tf.glorot_uniform_initializer())
b3 = tf.get_variable(name="b3", shape=[latent_dim], initializer=tf.zeros_initializer())
L3 = tf.matmul(L1, W3) + b3

z_mean = L2
z_log_var = L3
z = z_mean + tf.exp(0.5 * z_log_var) * tf.random_normal(shape=tf.shape(z_mean))

encoder = [z_mean, z_log_var, z]

# decoder
W4 = tf.get_variable(name="W4", shape=[latent_dim, 512], initializer=tf.keras.initializers.he_normal())
b4 = tf.get_variable(name="b4", shape=[512], initializer=tf.zeros_initializer())
L4 = tf.nn.relu(tf.matmul(Z, W4) + b4)

W5 = tf.get_variable(name="W5", shape=[512, input_dim], initializer=tf.glorot_uniform_initializer())
b5 = tf.get_variable(name="b5", shape=[input_dim], initializer=tf.zeros_initializer())
L5 = tf.nn.sigmoid(tf.matmul(L4, W5) + b5)

decoder = L5

# variational autoencoder
hypothesis = tf.nn.sigmoid(tf.matmul(tf.nn.relu(tf.matmul(z, W4) + b4), W5) + b5)

reconstruction_loss = (hypothesis - X) * (hypothesis - X) * input_dim
reguralization_loss = 0.5 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - z_log_var - 1, axis=-1, keepdims=True)

# cost & optimizer
cost = tf.reduce_mean(reguralization_loss + reconstruction_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
     
    for step in range(epochs):
        # shuffle data
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        
        loss = [0, 0]
        score = [0, 0]

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            
            c, _ = sess.run([cost, train], feed_dict={X: x_batch})
            loss[0] += c * len(x_batch)

        for i in range(0, x_test.shape[0], batch_size):
            x_batch = x_test[i:i + batch_size]
            
            c = sess.run(cost, feed_dict={X: x_batch})
            loss[1] += c * len(x_batch)
            
        print('loss: {:.4f} / {:.4f}\tstep {}  {:.2f} sec'.format(loss[0] / x_train.shape[0], loss[1] / x_test.shape[0], step + 1, time.time() - start_time))
        
    plot_results((encoder, decoder), (x_test, y_test), sess)
