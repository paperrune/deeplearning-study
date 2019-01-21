from keras.datasets import mnist
from keras.initializers import RandomUniform
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

batch_size = 128
epochs = 20
learning_rate = 0.5
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
y_train = to_categorical(y_train, num_classes)

x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(num_classes,
                activation='softmax',
                input_shape=(784,),
                kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=learning_rate),
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))