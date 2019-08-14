
from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math

val_datagen = ImageDataGenerator(rescale=1./255)

val_dir = os.path.join('../photo/scenario/test/')
val_generator = val_datagen.flow_from_directory(val_dir, batch_size=1, target_size=(78, 78), color_mode='grayscale')

model = load_model('saved_models/model_v2-1.h5')

#model.summary()

output = model.predict_generator(val_generator, steps=1)

#output[output>=0.5] =1 
#output[output<0.5] = 0


#f = open("../result.txt", 'w')

#output_l = output.tolist()
#output_str =  ','.join(output_l)
#f.write(output_str)

np.savetxt('../result.txt', output)

print(output)
print(val_generator.class_indices)
