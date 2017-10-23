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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
root_dir = os.path.abspath('..')

test_vols= 172*8

X,y = fetch_real_data('../data/sourcedata',10,8)

print(X.shape,y.shape)

#Best sim
model_saggital=   models.load_model('keras_logs/2017-10-13-13-50-06.epoch29-lossval0.21.hdf5')
#model_coronal = models.load_model('keras_logs/2017-10-17-10-22-15.epoch19-lossval0.16.hdf5')

#Best real:
#model_saggital=   models.load_model('keras_logs/real_2017-10-13-14-27-04.epoch29-lossval0.18.hdf5')
#model_coronal = models.load_model('keras_logs/real_2017-10-13-15-00-08.epoch29-lossval0.19.hdf5')

validation_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input_scaling,
    data_format='channels_last',)

num_slices = 30
validation_batch_size = 43
num_validation_steps = X_test.shape[0]/validation_batch_size
model_predictions_saggital = np.zeros((X_test.shape[0],num_slices))
y_test_fake = np.ones((X_test.shape[0]))

for i in range(num_slices):
    #saggital
    X_test_slice = preprocess_data_saggital(X,base_slice = 35+2*i)  
    #Redefine validation generator to reset
    validation_data_for_testing = validation_generator.flow(X_test_slice,y_test_fake,batch_size=validation_batch_size,shuffle=False)
    model_predictions_saggital[:,i] = model_saggital.predict_generator(validation_data_for_testing,num_validation_steps)[:,1]

    print('Slices complete:',i)

y_pred = np.mean(predictions_combined_sim,axis=1) > 0.84
print(classification_report((y!=0),y_pred))
print(confusion_matrix((y!=0),y_pred))