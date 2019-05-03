# # Only 2 lines will be added
# # Rest of the flow and code remains the same as default keras
# import plaidml.keras

# plaidml.keras.install_backend()

# Rest =====================

import os.path
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from image_data_generator import create_data_generator
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
import os
import imageio
import pathlib
import random
import tensorflow as tf
import tensorflow.keras as keras


TRAIN_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data1a/training'
TEST_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data1a/validation'

train_generator, validation_generator = create_data_generator(
    TRAIN_DATA_PATH, TEST_DATA_PATH)

# train_data = np.load('bottleneck_features_train.npy')
# # the features were saved in order, so recreating the labels is easy
# train_labels = np.array([0] * 920 + [1] * 920)

# validation_data = np.load('bottleneck_features_validation.npy')
# validation_labels = np.array([0] * 230 + [1] * 230)

BATCH_SIZE = 16

vgg = applications.VGG16(
    include_top=False, weights='imagenet', input_shape=(150, 150, 3))
vgg.trainable = False

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

filepath = "binary-transfer-vgg-{epoch:02d}-loss{val_loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit_generator(train_generator,
                    epochs=50,
                    validation_data=validation_generator,
                    callbacks=[checkpoint])
