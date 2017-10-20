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
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
from keras import models
from keras.callbacks import ModelCheckpoint
from time import gmtime, strftime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

root_dir = os.path.abspath('..')

def fetch_real_data(base_dir,num_subjects):
    '''Load in simulated data and motion files.'''
    subject_list = os.listdir((os.path.join(base_dir)))
    subject_list = [item for item in subject_list if item.startswith('sub') == True] #Filter everything but subjects
    subject_list = sorted(subject_list) #sort in numerical order to make OS independent
    counter = 0
    num_vols=172
    X = np.zeros((num_vols*num_subjects,128,128,64))
    y = np.zeros(num_vols*num_subjects)
    X_subject = np.zeros((128,128,64,num_vols))
    y_subject = np.zeros(num_vols)
    for subject_index, subject_number in enumerate(subject_list):
        if subject_index < num_subjects:
            data_path = os.path.join(base_dir,subject_number,'dwi.nii.gz')
            if os.path.isfile(data_path):
                data_header = nib.load(data_path)
                X_subject = data_header.get_data()
                y_subject = np.load(os.path.join(base_dir,subject_number,'y_manual.npy'))
                start_index = counter*num_vols
                end_index = (counter+1)*num_vols
                X[start_index:end_index,:] = np.moveaxis(X_subject,3,0)
                y[start_index:end_index] = y_subject
                counter += 1
    return X,y

def preprocess_data_coronal(X,target_height=299,target_width=299, base_slice=64,rescale=False,):
    '''Convert each MR volume to three slices through a single plane, scales the data and resamples
    to 299 by 299 pixels. Optionally performs augmentation.'''   
    #slices = [22,36,50] #Planes to slice
    slices = np.array([base_slice,base_slice,base_slice]) #Planes to slice
    pad_max = np.max([X.shape[1],X.shape[3]]) #Width to pad images to
    
    num_volumes = X.shape[0]
    height = X.shape[1]
    width = X.shape[3]
    num_slices = X.shape[2]
    X_preprocessed = np.zeros((num_volumes,target_height,target_width,3))

    for i in range(num_volumes):
        for j in range(3):
            if (j == 0):
                X_preprocessed[i,:,:,j] = imresize(pad_image(np.squeeze(X[i,:,slices[j],:]),pad_max),(target_width,target_height))
            if (j == 1):
                X_preprocessed[i,:,:,j] = imresize(pad_image(np.squeeze(X[i,:,slices[j],:]),pad_max),(target_width,target_height))
            if (j == 2):
                X_preprocessed[i,:,:,j] = imresize(pad_image(np.squeeze(X[i,:,slices[j],:]),pad_max),(target_width,target_height))     
   
    if rescale == True:
        X_preprocessed = X_preprocessed.astype(np.float32)
        X_preprocessed/= 255
        percentile =np.percentile(X_preprocessed,99.9)
        X_preprocessed[X_preprocessed>percentile] = percentile
        X_preprocessed/=percentile
        X_preprocessed -= 0.5
        X_preprocessed *= 2.
    return X_preprocessed

def preprocess_data_saggital(X,target_height=299,target_width=299, base_slice=64,rescale=False,):
    '''Convert each MR volume to three slices through a single plane, scales the data and resamples
    to 299 by 299 pixels. Optionally performs augmentation.'''   
    #slices = [22,36,50] #Planes to slice
    slices = np.array([base_slice,base_slice,base_slice]) #Planes to slice
    pad_max = np.max([X.shape[1],X.shape[3]]) #Width to pad images to
    
    num_volumes = X.shape[0]
    height = X.shape[1]
    width = X.shape[2]
    num_slices = X.shape[3]
    X_preprocessed = np.zeros((num_volumes,target_height,target_width,3))

    for i in range(num_volumes):
        for j in range(3):
            if (j == 0):
                X_preprocessed[i,:,:,j] = imresize(pad_image(np.squeeze(X[i,slices[j],:,:]),pad_max),(target_width,target_height))
            if (j == 1):
                X_preprocessed[i,:,:,j] = imresize(pad_image(np.squeeze(X[i,slices[j],:,:]),pad_max),(target_width,target_height))
            if (j == 2):
                X_preprocessed[i,:,:,j] = imresize(pad_image(np.squeeze(X[i,slices[j],:,:]),pad_max),(target_width,target_height))     
   
    if rescale == True:
        X_preprocessed = X_preprocessed.astype(np.float32)
        X_preprocessed/= 255
        percentile =np.percentile(X_preprocessed,99.9)
        X_preprocessed[X_preprocessed>percentile] = percentile
        X_preprocessed/=percentile
        X_preprocessed -= 0.5
        X_preprocessed *= 2.
    return X_preprocessed

def pad_image(image,pad_max):
    if pad_max == 0:
        return image
    else:
        pad_width = np.array([[pad_max,pad_max],[pad_max,pad_max]])-[image.shape,image.shape]
        pad_width=np.transpose(pad_width)
        pad_width[:,0] = np.floor(pad_width[:,0]/2)
        pad_width[:,1] = np.ceil(pad_width[:,1]/2)
        return np.lib.pad(image,pad_width,'constant',constant_values=(0))
    
def preprocess_input_scaling(x):
        x=x.astype(np.float32)
        x /= 255.
        percentile =np.percentile(x,99.9)
        x[x>percentile] = percentile
        x/=percentile
        x -= 0.5
        x *= 2.
        return x
    
def save_model(model,filepath):
    if not os.path.isfile(filepath):
        model.save(filepath)

def get_data_for_training(view,num_train=5,num_test=2):
    #view = 'coronal'
    #view = 'saggital'

    num_to_fetch = num_train + num_test
    X,y = fetch_real_data('../data/sourcedata/',num_to_fetch)
    print(X.shape,y.shape)

    num_vols = 172

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

    print('Train shape X',X_train.shape)
    print('Train shape y',X_train.shape)

    print('Test shape X',X_test.shape)
    print('Test shape y',y_test.shape)
    
    return X_train, X_test, y_train, y_test

def setup_model():
    #Set up inception v3 for transfer learning
    base_model = InceptionV3(weights='imagenet',include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(16,activation='relu')(x)
    predictions = Dense(2,activation='softmax')(x)

    model = Model(inputs=base_model.input,outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                     metrics=['accuracy'])
    return model

def train_model(X_train,X_test,y_train,y_test,model,num_epochs=30):
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
    now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    filename = 'keras_logs/real_'+now+'.epoch{epoch:02d}-lossval{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath=filename,
                            period=10)

    train_steps_per_epoch = train_examples/train_batch_size
    validation_steps_per_epoch = validation_examples/validation_batch_size
    print(train_steps_per_epoch)
    print(validation_steps_per_epoch)
    print('Model name:','real_keras_logs/'+now)
    
    history = model.fit_generator(generator = train_data,
                           steps_per_epoch=train_steps_per_epoch,
                           epochs = num_epochs,
                           validation_data = validation_data,
                           validation_steps = validation_steps_per_epoch,
                           class_weight=[1,5],
                            callbacks=[checkpoint])
    
    validation_data = validation_generator.flow(X_test,to_categorical(y_test,2),batch_size=validation_batch_size,shuffle=False)
    y_pred = model.predict_generator(validation_data,validation_steps_per_epoch)[:,1] > 0.5
    print(confusion_matrix(y_test,y_pred))
    return model

def test_model(X_test,y_test,model_coronal,model_saggital,num_slices):
    validation_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input_scaling,
    data_format='channels_last',)

    validation_batch_size = 43
    num_validation_steps = X_test.shape[0]/validation_batch_size
    model_predictions_coronal = np.zeros((X_test.shape[0],num_slices))
    model_predictions_saggital = np.zeros((X_test.shape[0],num_slices))
    y_test_fake = np.ones((X_test.shape[0]))

    for i in range(num_slices):
        #Coronal
        X_test_slice = preprocess_data_coronal(X_test[:,10:110,:,:],base_slice = 20+2*i)  
        #Redefine validation generator to reset
        validation_data_for_testing = validation_generator.flow(X_test_slice,y_test_fake,batch_size=validation_batch_size,shuffle=False)
        model_predictions_coronal[:,i] = model_coronal.predict_generator(validation_data_for_testing,num_validation_steps)[:,1]

        #saggital
        X_test_slice = preprocess_data_saggital(X_test,base_slice = 35+2*i)  
        #Redefine validation generator to reset
        validation_data_for_testing = validation_generator.flow(X_test_slice,y_test_fake,batch_size=validation_batch_size,shuffle=False)
        model_predictions_saggital[:,i] = model_saggital.predict_generator(validation_data_for_testing,num_validation_steps)[:,1]

        print('Slices complete:',i)
    predictions_combined = np.concatenate((model_predictions_coronal,model_predictions_saggital),axis=1)
    return(predictions_combined)

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
	    
run_test = True
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