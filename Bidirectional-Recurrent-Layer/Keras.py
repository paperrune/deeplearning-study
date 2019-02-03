import keras 
from keras.datasets import mnist
from keras.layers import Bidirectional, Dense, SimpleRNN
from keras.models import Sequential
from keras.optimizers import SGD
 
batch_size = 128
epochs = 100
learning_rate = 0.005
momentum = 0.9
time_step = 28
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], time_step, 784 // time_step).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.reshape(x_test.shape[0], time_step, 784 // time_step).astype('float32') / 255
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Bidirectional(SimpleRNN(128,
                                  activation='relu',
                                  kernel_initializer='he_normal'),
                        input_shape=(time_step, 784 // time_step)))
model.add(Dense(10,
                activation='softmax',
                kernel_initializer='glorot_uniform'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=learning_rate, momentum=momentum, nesterov=True),
              metrics=['accuracy'])
 
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))