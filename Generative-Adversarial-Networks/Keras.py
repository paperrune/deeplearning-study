import numpy as np

from keras.datasets import mnist
from keras.initializers import RandomUniform
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from PIL import Image
 
batch_size = 128
input_dim = 784
iteration = 100000
learning_rate = 0.0001
noise_dim = 100
 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape([x_train.shape[0], -1]).astype('float32') / 255

# generator
input = Input(shape=(noise_dim,))
layer = Dense(512,
              activation='relu',
              kernel_initializer='he_normal')(input)
layer = Dense(input_dim, activation='sigmoid')(layer)

generator = Model(input, layer, name='generator')
generator.summary()

# discriminator
input = Input(shape=(input_dim,))
layer = Dense(512,
              activation='relu',
              kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01))(input)
layer = Dense(1,
              activation='sigmoid',
              kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01))(layer)

discriminator = Model(input, layer, name='discriminator')
discriminator.summary()

# optimizer
input = [Input(shape=(noise_dim,)), Input(shape=(input_dim,))]
generator.trainable = False

fake_real_discriminator = Model(input, [discriminator(generator(input[0])), discriminator(input[1])])
fake_real_discriminator.compile(loss='binary_crossentropy',
                                optimizer=Adam(lr=learning_rate / 2),
                                metrics=['accuracy'])

discriminator.trainable = False
generator.trainable = True

fake_generator = Model(input[0], discriminator(generator(input[0])))
fake_generator.compile(loss='binary_crossentropy',
                       optimizer=Adam(lr=learning_rate),
                       metrics=['accuracy'])


for i in range(iteration):
    index = np.random.randint(0, len(x_train) - batch_size + 1)
    loss  = [None] * 2
    noise = np.random.uniform(-1, 1, size=(batch_size, noise_dim))

    x_batch = x_train[index:index + batch_size]
    loss[0] = fake_real_discriminator.train_on_batch([noise, x_batch], [np.array([0] * batch_size), np.array([1] * batch_size)])
    loss[1] = fake_generator.train_on_batch(noise, np.array([1] * batch_size))

    print(i + 1, 'D_fake:', [loss[0][1], loss[0][3]], '\tD_real:', [loss[0][2], loss[0][4]], '\tG:', loss[1])

    if i % 10000 == 9999:
        array = generator.predict(noise, batch_size=batch_size)
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
