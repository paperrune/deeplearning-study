import numpy as np

from keras.datasets import cifar10
from keras.initializers import RandomUniform
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Input, LeakyReLU, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam
from PIL import Image
 
batch_size = 128
input_dim = (32, 32, 3)
iteration = 100000
learning_rate = 0.0001
noise_dim = 100
 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape([x_train.shape[0]] + list(input_dim)).astype('float32') / 255
x_train = 2 * x_train - 1

# generator
generator = Sequential()
generator.add(Dense(4 * 4 * 256,
                    input_shape=(noise_dim,),
                    kernel_initializer='he_normal'))
generator.add(BatchNormalization())
generator.add(Activation('relu'))
generator.add(Reshape((4, 4, 256)))
# 4, 4, 256 /

generator.add(Conv2DTranspose(128,
                              kernel_initializer='he_normal',
                              kernel_size=(5, 5),
                              padding='same',
                              strides=(2, 2)))
generator.add(BatchNormalization())
generator.add(Activation('relu'))
# 8, 8, 128 /

generator.add(Conv2DTranspose(64,
                              kernel_initializer='he_normal',
                              kernel_size=(5, 5),
                              padding='same',
                              strides=(2, 2)))
generator.add(BatchNormalization())
generator.add(Activation('relu'))
# 16, 16, 64 /

generator.add(Conv2DTranspose(3,
                              activation='tanh',
                              kernel_size=(5, 5),
                              padding='same',
                              strides=(2, 2)))
# 32, 32, 3 /

generator.summary()

# discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64,
                         input_shape=input_dim,
                         kernel_initializer=RandomUniform(-0.01, 0.01),
                         kernel_size=(5, 5),
                         padding='same',
                         strides=(2, 2)))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))
# 16, 16, 64 /

discriminator.add(Conv2D(128,
                         kernel_initializer=RandomUniform(-0.01, 0.01),
                         kernel_size=(5, 5),
                         padding='same',
                         strides=(2, 2)))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))
# 8, 8, 128 /

discriminator.add(Conv2D(256,
                         kernel_initializer=RandomUniform(-0.01, 0.01),
                         kernel_size=(5, 5),
                         padding='same',
                         strides=(2, 2)))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))
# 4, 4, 256 /

discriminator.add(Flatten())
discriminator.add(Dense(1,
                        activation='sigmoid',
                        kernel_initializer=RandomUniform(-0.01, 0.01)))

discriminator.summary()

# optimizer
input = [Input(shape=(noise_dim,)), Input(shape=input_dim)]
generator.trainable = False

fake_real_discriminator = Model(input, [discriminator(generator(input[0])), discriminator(input[1])])
fake_real_discriminator.compile(loss='binary_crossentropy',
                                optimizer=Adam(lr=learning_rate),
                                metrics=['accuracy'])

discriminator.trainable = False
generator.trainable = True

fake_generator = Model(input[0], discriminator(generator(input[0])))
fake_generator.compile(loss='binary_crossentropy',
                       optimizer=Adam(lr=learning_rate),
                       metrics=['accuracy'])


for i in range(iteration):
    index = (i * batch_size) % (len(x_train) - batch_size)
    loss  = [None] * 2
    noise = np.random.uniform(-1, 1, size=(batch_size, noise_dim))

    x_batch = x_train[index:index + batch_size]
    loss[0] = fake_real_discriminator.train_on_batch([noise, x_batch], [np.array([0] * batch_size), np.array([1] * batch_size)])
    loss[1] = fake_generator.train_on_batch(noise, np.array([1] * batch_size))

    print(i, 'D_fake:', [loss[0][1], loss[0][3]], '\tD_real:', [loss[0][2], loss[0][4]], '\tG:', loss[1])

    if i % 10000 == 9999:
        array = generator.predict(noise, batch_size=batch_size)
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
