
import numpy as np
import tensorflow as tf
import keras
import pandas as pd 
import os
import cv2
from keras.models import load_model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Add, Flatten
from keras.models import Model,Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import models
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"]="1"

modelCheckpoint = ModelCheckpoint('nnfl_64_200.hdf5', save_best_only=True)

classifier = Sequential()

# Encoder Layers
classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(150,150,3)))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2), padding='same'))

classifier.add(Flatten())
classifier.add(Dense(512, activation = 'relu'))
classifier.add(Dense(6, activation = 'softmax'))

classifier.summary()

datagen = ImageDataGenerator(
	rescale=1./255,
    validation_split=0.05,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    
	)

classifier.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

Train_images = datagen.flow_from_directory('/home_01/f20160007/nnfl/seg_train', classes=['buildings','forest','glacier','sea','mountain','street'], target_size=(150,150), batch_size=64, class_mode='categorical', shuffle=True, seed=13, subset="training")

val_generator = datagen.flow_from_directory('/home_01/f20160007/nnfl/seg_train', classes=['buildings','forest','glacier','sea','mountain','street'], target_size=(150,150), batch_size=64, class_mode='categorical', shuffle=True, seed=13, subset="validation")

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
classifier_weights = classifier.fit_generator(Train_images, steps_per_epoch=150,epochs=200,validation_steps=50,validation_data=(val_generator),callbacks=[modelCheckpoint,es])
# classifier.save_model('autoencoder_classification1.h5')
