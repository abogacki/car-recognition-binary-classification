import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import os
import os.path


def save_bottlebeck_features(train_data_dir, validation_data_dir, image_size=(150, 150), batch_size=16, features_path='bottleneck_features'):
    datagen = ImageDataGenerator(rescale=1. / 255)

    nb_train_samples = len([name for name in os.listdir(
        train_data_dir) if os.path.isfile(os.path.join(train_data_dir, name))])

    nb_validation_samples = len([name for name in os.listdir(validation_data_dir)
                                 if os.path.isfile(os.path.join(validation_data_dir, name))])

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)

    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


TRAIN_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data1a/training/'
TEST_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data1a/validation/'

save_bottlebeck_features(TRAIN_DATA_PATH, TEST_DATA_PATH)
