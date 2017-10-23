from library import *

import os
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imresize, imrotate

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.utils import to_categorical
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
from keras import models
from keras.callbacks import ModelCheckpoint

from time import gmtime, strftime

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, f1_score
root_dir = os.path.abspath('..')
set_gpu_usage_fraction(0.5)
#view = 'coronal'
view = 'saggital'


X,y = fetch_real_data('../data/sourcedata/',7)
print(X.shape,y.shape)

num_vols = 172
num_train = 5
num_test = 2
slices_to_extract = [50,64,80]

X_train = np.zeros((num_vols*num_train*len(slices_to_extract),299,299,3))
X_test = np.zeros((num_vols*num_test*len(slices_to_extract),299,299,3))
y_train = np.zeros((num_vols*num_train*len(slices_to_extract)))
y_test = np.zeros((num_vols*num_test*len(slices_to_extract)))

for slice_num, slice_indx in enumerate(slices_to_extract):
    start_indx_train = slice_num * num_vols*num_train
    end_indx_train = (slice_num+1) * num_vols*num_train
    start_indx_test = slice_num * num_vols*num_test
    end_indx_test = (slice_num+1) * num_vols*num_test
    if view == 'saggital':
        X_preprocessed = preprocess_data_saggital(X,base_slice=slice_indx)
    elif view == 'coronal':
        X_preprocessed = preprocess_data_coronal(X,base_slice=slice_indx)
    else:
        print('View not recognised')    
    X_train[start_indx_train:end_indx_train,:] = X_preprocessed[:num_vols*num_train]
    X_test[start_indx_test:end_indx_test,:] = X_preprocessed[num_vols*num_train:]
    y_train[start_indx_train:end_indx_train]= y[:num_vols*num_train]
    y_test[start_indx_test:end_indx_test]=y[num_vols*num_train:]

y_train = (y_train!=0)
y_test = (y_test!=0)

#Free up memory
del X,y

print('Train shape X',X_train.shape)
print('Train shape y',X_train.shape)
print('Test shape X',X_test.shape)
print('Test shape y',y_test.shape)


#Set up for training
train_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input_scaling,
    data_format='channels_last')

validation_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input_scaling,
    data_format='channels_last',)

train_batch_size = 43
validation_batch_size = 43
train_examples = X_train.shape[0]
validation_examples = X_test.shape[0]
train_data = train_generator.flow(X_train,to_categorical(y_train,2),batch_size=train_batch_size,shuffle=True)
validation_data = validation_generator.flow(X_test,to_categorical(y_test,2),batch_size=validation_batch_size,shuffle=False)

model = setup_model()
#model_trained = train_model(X_train,X_test,y_train,y_test,model,num_epochs=30,train_batch_size=train_batch_size,validation_batch_size=validation_batch_size)

del X_train,X_test

X,y = fetch_real_data('../data/sourcedata/',2)

#Find optimal threshold to maximise f1 score
model_trained = models.load_model('keras_logs/2017-10-23-14-32-31.epoch29-lossval0.20.hdf5')
predictions = test_model(X,y,model_trained,model_trained,30)
best_score = 0
best_thresh = 0
for threshold in range(10,90,1):
	thresh  = threshold / 100
	y_pred = np.mean(predictions,axis=1) > thresh
	score = f1_score(y!=0,y_pred)
	if score > best_score:
		best_score = score
		best_thresh = thresh

print('Best f1 score:',best_score)
print('Best threshold:',best_thresh)
y_pred_best = np.mean(predictions,axis=1) > best_thresh
print(classification_report((y!=0),y_pred_best))
print(confusion_matrix((y!=0),y_pred_best))

