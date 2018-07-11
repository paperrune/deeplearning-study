import keras 
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Bidirectional, SimpleRNN
from keras.models import Sequential
from keras.optimizers import SGD
 
batch_size = 128
epochs = 100
num_classes = 10
time_step = 28
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, time_step, 784 // time_step)
x_test = x_test.reshape(10000, time_step, 784 // time_step)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes) 

model = Sequential()
model.add(Bidirectional(SimpleRNN(128,
                                  activation='relu'),
                        input_shape=(time_step, 784 // time_step)))
model.add(Dense(num_classes,
                activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.03, decay=1e-6),
              metrics=['accuracy'])
 
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))
