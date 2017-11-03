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
from sklearn.metrics import confusion_matrix, recall_score,accuracy_score, precision_score, classification_report, f1_score
root_dir = os.path.abspath('..')

set_gpu_usage_fraction(0.5)

mode='three_class'
#view = 'coronal'
view = 'saggital'

num_vols = 172

X, y = fetch_sim_data(os.path.join(root_dir,'data/sims-moremotion-dhcp'),5,mode=mode,translation_threshold=2.5, rotation_threshold = 2.5,volumes_per_subject=num_vols)

print('Volumes with no motion:',np.sum(y==0))
print('Volumes with severe motion:',np.sum(y==1))
print('Volumes with moderate motion:',np.sum(y==2))

#Load in training and validation data
num_train = 5
num_test = 0
slices_to_extract = [20,36,52]

X_train = np.zeros((num_vols*num_train*len(slices_to_extract),299,299,3))
X_test = np.zeros((num_vols*num_test*len(slices_to_extract),299,299,3))
y_train = np.zeros((num_vols*num_train*len(slices_to_extract)))
y_test = np.zeros((num_vols*num_test*len(slices_to_extract)))

for slice_num, slice_indx in enumerate(slices_to_extract):
    start_indx_train = slice_num * num_vols*num_train
    end_indx_train = (slice_num+1) * num_vols*num_train
    start_indx_test = slice_num * num_vols*num_test
    end_indx_test = (slice_num+1) * num_vols*num_test
    if view == 'coronal':
        X_preprocessed = preprocess_data_coronal(X,base_slice=slice_indx)
    elif view == 'saggital':
        X_preprocessed = preprocess_data_saggital(X,base_slice=slice_indx)
    else:
        print('View not recognised')

    X_train[start_indx_train:end_indx_train,:] = X_preprocessed[:num_vols*num_train]
    X_test[start_indx_test:end_indx_test,:] = X_preprocessed[num_vols*num_train:]
    y_train[start_indx_train:end_indx_train]= y[:num_vols*num_train]
    y_test[start_indx_test:end_indx_test]=y[num_vols*num_train:]

if mode=='three_class':
    X_train = np.delete(X_train,np.where(y_train==2),0)
    y_train = np.delete(y_train,np.where(y_train==2),0)
    X_test = np.delete(X_test,np.where(y_test==2),0)
    y_test = np.delete(y_test,np.where(y_test==2),0)
    
#Clear memory
del X,y, X_preprocessed

print('Size of X train is:',X_train.shape)
print('Size of y train is:',y_train.shape)
print('Size of X test is:',X_test.shape)
print('Size of y test is:',y_test.shape)
print('Percentage motion-free in train set:',1-np.sum(y_train)/len(y_train))
print('Number with motion in train set:',np.sum(y_train))
print('Percentage motion-free in test set:',1-np.sum(y_test)/len(y_test))
print('Number with motion in test set:',np.sum(y_test))

#Set up for training
train_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input_scaling,
    data_format='channels_last')

validation_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input_scaling,
    data_format='channels_last',)

train_batch_size = 54
validation_batch_size = 54
train_examples = X_train.shape[0]
validation_examples = X_test.shape[0]
train_data = train_generator.flow(X_train,to_categorical(y_train,2),batch_size=train_batch_size,shuffle=True)
validation_data = validation_generator.flow(X_test,to_categorical(y_test,2),batch_size=validation_batch_size,shuffle=False)

train_now = True
if train_now == True:
	model = setup_model()
	model_trained = train_model(X_train,X_test,y_train,y_test,model,num_epochs=30,train_batch_size=train_batch_size,validation_batch_size=validation_batch_size)
if train_now !=True:
        model_trained = models.load_model('keras_logs/2017-10-23-14-12-08.epoch29-lossval0.21.hdf5')
#save_model(model_trained_saggital,('keras_logs/saggital_'+str(i)+'.h5'))

#Find optimal threshold to maximise f1 score
num_datasets = 5
X,y = fetch_real_data('../data/sourcedata/',num_datasets,1)
predictions = test_model(X,y,model_trained,model_trained,30)

#Try for several different datasets
for j in range(num_datasets):
	print('Dataset:',j)
	prediction_set = predictions[172*j:172*(j+1),:]
	y_set = y[172*j:172*(j+1)]
	best_score = 0
	best_thresh = 0
	for threshold in range(10,90,1):
		thresh  = threshold / 100
		y_pred = np.mean(prediction_set,axis=1) > thresh 
		score = f1_score(y_set!=0,y_pred)
		score_recall = recall_score(y_set!=0,y_pred)
		if score > 0.9:
			print(thresh,score)
		if score > best_score:
			best_score = score
			best_thresh = thresh

	print('Best f1 score:',best_score)
	print('Best threshold:',best_thresh)
	y_pred_best = np.mean(prediction_set,axis=1) > best_thresh
	print(classification_report((y_set!=0),y_pred_best))
	print(confusion_matrix((y_set!=0),y_pred_best))
