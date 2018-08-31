#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 21:15:40 2018

@author: shubhamsinha
"""
import matplotlib.pyplot as plt
import numpy
#Keras model can be sequential or graphical
from keras.models import Sequential

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initialize the CNN
classifier= Sequential()

#convolution
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))

#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#2nd conv
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#flatten
classifier.add(Flatten())

#fully connected
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(p=0.6))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(p=0.5))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(p=0.3))

#classifier.add(Dense(units=64, activation='relu'))
#classifier.add(Dropout(p=0.5))


classifier.add(Dense(units=1,  activation='sigmoid'))

#compiling the whole cnn
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting cnn to images
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
]

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

history=classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=100,
        validation_data=test_set,
        validation_steps = 2000/32,
        callbacks=callbacks)

#test_single=test_datagen.flow_from_directory(
#        'dataset/single_prediction',
#        target_size=(64, 64),
#        batch_size=32,
#        class_mode='binary')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.show()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.figure()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


