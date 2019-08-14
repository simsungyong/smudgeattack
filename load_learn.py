import keras
from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add

import os
import matplotlib.pyplot as plt
import numpy as np
import math
from keras import preprocessing
from utils import *
from keras import backend as k

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join('../photo_2/3_1/training')
val_dir = os.path.join('../photo_2/3_1/validation')

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=14, target_size=(78, 78), color_mode='grayscale', class_mode='multi_categorical')
val_generator = val_datagen.flow_from_directory(val_dir, batch_size=14, target_size=(78, 78), color_mode='grayscale', class_mode='multi_categorical')


model = load_model('saved_models/model_v2.h5')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator,
                        steps_per_epoch=120,
                        epochs=20,
                        validation_data=val_generator,
                        validation_steps=30)


model.save('saved_models/model_v2-2.h5')
