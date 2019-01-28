from keras.datasets import mnist
from keras.initializers import RandomUniform
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

# input image dimensions
img_rows, img_cols = 28, 28

batch_size = 128
epochs = 30
learning_rate = 0.1
momentum = 0.9
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape([x_train.shape[0], img_rows, img_cols, 1]).astype('float32') / 255
y_train = to_categorical(y_train, num_classes)

x_test = x_test.reshape([x_test.shape[0], img_rows, img_cols, 1]).astype('float32') / 255
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(24,
                 activation='relu',                 
                 kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1),
                 kernel_size=(5, 5),
                 strides=(2, 2),
                 input_shape=(img_rows, img_cols, 1)))
model.add(Flatten())
model.add(Dense(512,
                activation='relu',
                kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)))
model.add(Dense(num_classes,
                activation='softmax',
                kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01)))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=learning_rate, momentum=momentum, nesterov=True),
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))