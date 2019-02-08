from keras.datasets import mnist
from keras.initializers import RandomUniform
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

batch_size = 128
epochs = 20
learning_rate = 0.5

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape([x_train.shape[0], -1]).astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)

x_test = x_test.reshape([x_test.shape[0], -1]).astype('float32') / 255
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Dense(10,
                activation='sigmoid',
                input_shape=(784,),
                kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)))
model.summary()
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=learning_rate),
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
