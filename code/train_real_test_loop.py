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

run_train = False
if run_train == True:
	#Train
	X_train_saggital, X_val_saggital, y_train_saggital, y_val_saggital = get_data_for_training('saggital')
	X_train_coronal, X_val_coronal, y_train_coronal, y_val_coronal = get_data_for_training('coronal')


	print('Baseline performance for val data:',1-np.sum(y_val_coronal)/np.shape(y_val_coronal))


	num_steps=5
	num_slices = 30


	for i in range(2,num_steps+1):
	    model = setup_model()
	    model_trained_saggital = train_model(X_train_saggital[:172*3*i],X_val_saggital,y_train_saggital[:172*3*i],y_val_saggital,model,20)
	    save_model(model_trained_saggital,('keras_logs/saggital_'+str(i)+'.h5'))
	    model = setup_model()
	    model_trained_coronal = train_model(X_train_coronal[:172*3*i],X_val_coronal,y_train_coronal[:172*3*i],y_val_coronal,model,20)
	    save_model(model_trained_coronal,('keras_logs/coronal_'+str(i)+'.h5'))
	    
run_test = False
if run_test ==  True:
	#Test
	#Fetch test data
	print('Fetching data')
	X_test,y_test = fetch_real_data('../data/sourcedata/test/',3)

	num_steps=4
	num_slices = 30

	predictions_combined_all=np.zeros((num_steps,X_test.shape[0],num_slices*2))

	for i in range(3,num_steps+1):
	    model_trained_saggital = models.load_model('../data_old/keras_logs/saggital_'+str(i)+'.h5')
	    model_trained_coronal = models.load_model('../data_old/keras_logs/coronal_'+str(i)+'.h5')
	    
	    predictions_combined_all[i-1,:,:] = test_model(X_test,y_test,model_trained_coronal,model_trained_saggital,num_slices)
	    predictions_combined = np.squeeze(predictions_combined_all[i-1,:,:])
	    y_pred = np.mean(predictions_combined,axis=1)>0.5
	    print(i)
	    print(classification_report(y_test!=0,y_pred))
	    print(confusion_matrix(y_test!=0,y_pred))
	np.save(predictions_combined_all,'predictions_combined_all.npy')

test_on_sim_data = True
if test_on_sim_data == True:
    X,y = fetch_real_data('../data/sourcedata/',1)
    num_slices = 30
    model_trained_saggital = models.load_model('keras_logs/saggital_1.h5')
    model_trained_coronal = model_trained_saggital
    predictions = test_model(X,y,model_trained_coronal,model_trained_saggital,num_slices)
    for threshold in range(0.05,0.9,0.05):
        print(threshold)
        y_pred = np.mean(predictions,axis=1)>threshold
        print(classification_report(y!=0,y_pred))
        print(confusion_matrix(y!=0,y_pred))
