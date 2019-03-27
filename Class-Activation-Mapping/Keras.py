import cv2
import numpy as np
import os

from keras.datasets import cifar10
from keras.layers import Activation, BatchNormalization, Conv2D, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

batch_size = 128
epochs = 50
input_shape = (32, 32, 3)
learning_rate = 0.1
momentum = 0.9

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape([x_train.shape[0]] + list(input_shape)).astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)

x_test = x_test.reshape([x_test.shape[0]] + list(input_shape)).astype('float32') / 255
y_test = to_categorical(y_test, num_classes=10)

input = Input(shape=input_shape)
layer = Conv2D(32,
               kernel_initializer='he_normal',
               kernel_size=(3, 3),
               padding='same')(input)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Conv2D(32,
               kernel_initializer='he_normal',
               kernel_size=(3, 3),
               padding='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = MaxPooling2D()(layer)
layer = Conv2D(64,
               kernel_initializer='he_normal',
               kernel_size=(3, 3),
               padding='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Conv2D(64,
               kernel_initializer='he_normal',
               kernel_size=(3, 3),
               padding='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = MaxPooling2D()(layer)
layer = Conv2D(128,
               kernel_initializer='he_normal',
               kernel_size=(3, 3),
               padding='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Conv2D(128,
               kernel_initializer='he_normal',
               kernel_size=(3, 3),
               padding='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Conv2D(10, kernel_size=(1, 1), name='class_activation_map')(layer)
layer = GlobalAveragePooling2D()(layer)
layer = Activation('softmax')(layer)

model = Model(input, layer)
model.summary()

generator = ImageDataGenerator(horizontal_flip=True, zoom_range=0.5)
generator.fit(x_train)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=learning_rate, momentum=momentum, nesterov=True),
              metrics=['accuracy'])

model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    steps_per_epoch=len(x_train) // batch_size + 1,
                    validation_data=(x_test, y_test))

model = Model(model.input, [model.output, model.get_layer('class_activation_map').output])
score, map = model.predict(x_test, batch_size=batch_size)

for i in range(10):
    if not os.path.exists('{}'.format(i)):
        os.makedirs('{}'.format(i))

for i in range(len(x_test)):
    CAM = map[i][:, :, np.argmax(score[i])]
    max = np.max(CAM)
    min = np.min(CAM)
    CAM = cv2.applyColorMap(np.uint8((CAM - min) / (max - min) * 255), cv2.COLORMAP_JET)    
    CAM = cv2.resize(CAM, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
    
    cv2.imwrite('{0}/{1:05d}.png'.format(np.argmax(score[i]), i), 0.5 * np.array(CAM) + 0.5 * np.uint8(x_test[i] * 255))
