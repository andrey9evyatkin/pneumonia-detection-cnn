from keras.preprocessing.image import ImageDataGenerator

from constants import *


def generate_data(data_type):
    data_generator = ImageDataGenerator(rescale=1./255)
    path = DATA_PATH + get_data_type_path(data_type)
    if data_type == 'train':
        data_generator = ImageDataGenerator(
            rescale=1/255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2)
    return data_generator.flow_from_directory(path, target_size=(224, 224), batch_size=16, class_mode='binary', shuffle=True)


def get_data_type_path(data_type):
    if data_type == 'test':
        return TEST_DATA_PATH
    if data_type == 'train':
        return TRAIN_DATA_PATH
    if data_type == 'val':
        return VAL_DATA_PATH
