import numpy as np
import tensorflow as tf
import keras
import pandas as pd 
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Add, Flatten
from keras.models import Model,Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import models
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model=models.load_model('./nnfl_64_200_3.hdf5')


test_data_gen_args = dict(rescale=1./255)
test_image_datagen = ImageDataGenerator(**test_data_gen_args)


test_generator = test_image_datagen.flow_from_directory('./seg_test/seg_test/', classes=['buildings','forest','glacier','sea','mountain','street'], target_size=(150,150), batch_size=1, class_mode='categorical', shuffle=False, seed=42)

print(model.evaluate_generator(test_generator, steps=3000))


test_generator.reset() # so that the outputs wont be in a weird order
predictions = model.predict_generator(test_generator, steps=3000)  
pred_class = np.argmax(predictions, axis=-1) #multiple categories


label_map = (test_generator.class_indices)
# print(label_map)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
predictions_label = [label_map[k] for k in pred_class]
test_classes=[label_map[k] for k in test_generator.classes]
# print(predictions_label, len(predictions_label))

filenames=test_generator.filenames
# print(len(filenames) , len(predictions_label))
  

true_classes = test_generator.classes                 #y_true
class_labels = list(test_generator.class_indices.keys())  


report = metrics.classification_report(true_classes, pred_class, target_names=class_labels)
print(report)  # prints precision, recall, F1 score for each class

confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=pred_class) 
print(confusion_matrix)

correct = np.where(pred_class==test_generator.classes)[0]
# print("correct labels" , len(correct),correct)

fig=plt.figure(figsize=(10,10))
for i, correct in enumerate(correct[:9]):
	
	ax=fig.add_subplot(3,3,i+1)
	ax.imshow(test_generator[0][0][correct], interpolation='none')
	ax.axis('off')
	ax.set_title("Pred- {}, Class- {}".format(predictions_label[correct],test_classes[correct]))
	plt.tight_layout()
plt.show()

incorrect = np.where(pred_class!=test_generator.classes)[0]
# print("incorrect labels" , len(incorrect),incorrect)

fig=plt.figure(figsize=(10,10))

for i, incorrect in enumerate(incorrect[:4]):
	
	ax=fig.add_subplot(2,2,i+1)
	ax.imshow(test_generator[0][0][incorrect], interpolation='none')
	ax.axis('off')
	ax.set_title("Pred- {}, Class- {}".format(predictions_label[incorrect],test_classes[incorrect]))
	plt.tight_layout()
plt.show()
