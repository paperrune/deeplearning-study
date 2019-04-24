import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Input, Lambda
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import Adam
 
batch_size = 128
epochs = 30
input_dim = 784
latent_dim = 2
learning_rate = 0.001
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
x_train = x_train.reshape([x_train.shape[0], -1]).astype('float32') / 255 
x_test = x_test.reshape([x_test.shape[0], -1]).astype('float32') / 255

# encoder
input = Input(shape=(input_dim,))
layer = Dense(512,
              activation='relu',
              kernel_initializer='he_normal')(input)
z_mean = Dense(latent_dim, name='z_mean')(layer)
z_log_var = Dense(latent_dim, name='z_variance')(layer)
z = Lambda(lambda x: x[0] + K.exp(0.5 * x[1]) * K.random_normal(shape=K.shape(x[0])), name='z')([z_mean, z_log_var])

encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# decoder
input = Input(shape=(latent_dim,))
layer = Dense(512,
              activation='relu',
              kernel_initializer='he_normal')(input)
layer = Dense(input_dim, activation='sigmoid')(layer)

decoder = Model(input, layer, name='decoder')
decoder.summary()

# variational autoencoder
VAE = Model(encoder.input, decoder(encoder(encoder.input)[2]))

reconstruction_loss = mean_squared_error(VAE.input, VAE.output) * input_dim
regularization_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - z_log_var - 1, axis=-1) * latent_dim

VAE.add_loss(K.mean(regularization_loss + reconstruction_loss))
VAE.compile(optimizer=Adam(lr=learning_rate))
VAE.summary()

VAE.fit(x_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, None))

# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
def plot_results(models, data, batch_size):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
    """

    encoder, decoder = models
    x_test, y_test = data

    filename = 'vae_mean.png'
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
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
            x_decoded = decoder.predict(z_sample)
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
    
plot_results((encoder, decoder), (x_test, y_test), batch_size)
