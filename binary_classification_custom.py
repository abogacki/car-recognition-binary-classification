# # Only 2 lines will be added
# # Rest of the flow and code remains the same as default keras
# import plaidml.keras

# plaidml.keras.install_backend()

# Rest =====================

from PIL import Image
import numpy as np
import os
import imageio
import pathlib
import random
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from image_data_generator import create_data_generator

TRAIN_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data1a/training'
TEST_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data1a/validation'

train_generator, validation_generator = create_data_generator(
    TRAIN_DATA_PATH, TEST_DATA_PATH)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(2, 2))

model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(2, 2))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

BATCH_SIZE = 16


filepath = "binary-custom-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // BATCH_SIZE,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // BATCH_SIZE,
    callbacks=[checkpoint]
)
