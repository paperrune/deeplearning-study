from keras.datasets import mnist
from keras.layers import Conv2D, Dense, DepthwiseConv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
 
# input image dimensions
img_rows, img_cols = 28, 28
 
batch_size = 128
epochs = 30
learning_rate = 0.05
momentum = 0.9
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
x_train = x_train.reshape([x_train.shape[0], img_rows, img_cols, 1]).astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
 
x_test = x_test.reshape([x_test.shape[0], img_rows, img_cols, 1]).astype('float32') / 255
y_test = to_categorical(y_test, num_classes=10)
 
model = Sequential()
model.add(Conv2D(24,
                 activation='relu',
                 kernel_initializer='he_normal',
                 kernel_size=(5, 5),
                 input_shape=(img_rows, img_cols, 1)))
model.add(MaxPooling2D())

# depthwise
model.add(DepthwiseConv2D(activation='relu',
                          depthwise_initializer='he_normal',
                          kernel_size=(5, 5)))

# pointwise
model.add(Conv2D(48,
                 activation='relu',
                 kernel_initializer='he_normal',
                 kernel_size=(1, 1)))

model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512,
                activation='relu',
                kernel_initializer='he_normal'))
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
