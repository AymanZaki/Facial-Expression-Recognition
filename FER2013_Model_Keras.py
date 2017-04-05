from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from FER2013_Input import FER2013_Input
import csv
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy import array
from scipy.misc import toimage
from resizeimage import resizeimage

batch_size = 128
num_classes = 7
epochs = 100

img_rows, img_cols = 42,42
fer = FER2013_Input('/home/alaa/Desktop/GP/')
Training_labels, Training_Images = fer.FER2013_Training_Set()
Testing_labels, Testing_Images = fer.FER2013_Testing_Set()
Validation_labels, Validation_Images = fer.FER2013_Validation_Set()
Training_Images = Training_Images.reshape(Training_Images.shape[0], img_rows, img_cols, 1)
Validation_Images = Validation_Images.reshape(Validation_Images.shape[0], img_rows, img_cols, 1)
Testing_Images = Testing_Images.reshape(Testing_Images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

Training_Images = Training_Images.astype('float32')
Validation_Images = Validation_Images.astype('float32')
Testing_Images = Testing_Images.astype('float32')
Training_Images /= 255
Validation_Images/=255
Testing_Images /= 255

Training_labels = keras.utils.to_categorical(Training_labels, num_classes)
Validation_labels = keras.utils.to_categorical(Validation_labels, num_classes)
Testing_labels = keras.utils.to_categorical(Testing_labels, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(3072, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(Training_Images, Training_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(Validation_Images, Validation_labels))
score = model.evaluate(Testing_Images, Testing_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])