
import numpy as np
import pylab as plt
import pandas as pd

from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

'''
PARAMETERS to change so as to improve the training

'''

#We can modify batch size and epochs to adjust improve the training
batch_size=32
epochs=50
vggsp500loss='categorical_crossentropy'  ##https://keras.io/api/losses/
vggsp500optimizer='rmsprop'             ##https://keras.io/api/optimizers/
vggsp500metrics=['accuracy']            ##https://keras.io/api/metrics/
'''
UTILITY FUNCTIONS

'''

def change_X_df__nparray_image(df_X_train_image_flattened ):
  '''
  setup_input_NN_image returns a dataframe of flaten image for x train and xtest
  then this function will change each date into a nparray list of images with 32, 32, 3 size 
  '''
  X_train_image=df_X_train_image_flattened
  nb_train=len(X_train_image.index)
  
  x_train=np.zeros((nb_train,32,32,3))
  for i in range(nb_train):
    tmp=np.array(X_train_image.iloc[i])
    tmp=tmp.reshape(32,32,3)
    x_train[i]=tmp
  return x_train

'''
PART 2 LOADING TRAINING DATAS AND FORMAT IT TO BE ACCEPTED BY THE NN
'''
#recuperation of datas 
X_train_image=pd.read_csv('datas/X_train_image.csv')
Y_train_StateClass_image=pd.read_csv('datas/Y_train_StateClass_image.csv')
Y_train_FutPredict_image=pd.read_csv('datas/Y_train_FutPredict_image.csv')


#setting up the index to Date
X_train_image=X_train_image.set_index("Date")
Y_train_StateClass_image=Y_train_StateClass_image.set_index("Date")
Y_train_FutPredict_image=Y_train_FutPredict_image.set_index("Date")

#modify dataset to np array for input to NN
x_train=change_X_df__nparray_image(X_train_image)
y_train_state=np.array(Y_train_StateClass_image)
y_train_value=np.array(Y_train_FutPredict_image)

##Setting up xtrain and ytrain
#Here we focus on predicting the future state Y_train_StateClass_image
nb_train=len(X_train_image.index)
x_train=np.zeros((nb_train,32,32,3))
for i in range(nb_train):
  tmp=np.array(X_train_image.iloc[i])
  tmp=tmp.reshape(32,32,3)
  x_train[i]=tmp
  
y_train=np.array(Y_train_StateClass_image)
#y_train=np.array(Y_train_FutPredict_image)

nb_train=len(X_train_image.index)
x_train=np.zeros((nb_train,32,32,3))
for i in range(nb_train):
  tmp=np.array(X_train_image.iloc[i])
  tmp=tmp.reshape(32,32,3)
  x_train[i]=tmp

y_train=np.array(Y_train_StateClass_image)
#y_train=np.array(Y_train_FutPredict_image)


#At this stage we have x and y to train the model

'''
PART 3 VGGSP500 TRAINING AND SAVING
we suppose that we have loaded xtrain and ytrain
This part is based on the Design of the NN
Her we find the Vgg16 quite usefull
'''

#Importing the VGG16 model
from keras.applications.vgg16 import VGG16, preprocess_input


#In our example we need to y into categorical as it has 6 categories
nb_classes=6
y_train = np_utils.to_categorical(y_train, nb_classes)


#Loading the VGG16 model with pre-trained ImageNet weights
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg_model.trainable = False # remove if you want to retrain vgg weights

vgg_model.summary()


##Transfert model from vgg
transfer_model = Sequential()
transfer_model.add(vgg_model)
transfer_model.add(Flatten())
transfer_model.add(Dense(128, activation='relu'))
transfer_model.add(Dropout(0.2))
transfer_model.add(Dense(6, activation='softmax'))

##Display summary of neural network
transfer_model.summary()

transfer_model.compile(loss=vggsp500loss, optimizer=vggsp500optimizer,
              metrics=vggsp500metrics)

##Saving the best model for each parameters
checkpoint = ModelCheckpoint("model/best_model"+vggsp500loss+"_"+vggsp500optimizer+"_"+batch_size.hdf5", \
                                monitor='loss', verbose=1, \
                                save_best_only=True, mode='auto', period=1)

##Fitting the model on the train data and labels.
history = transfer_model.fit(x_train, y_train, \
                              batch_size=batch_size, epochs=epochs, \
                              validation_split=0.2, verbose=1, shuffle=True, \
                              callbacks=[checkpoint])

# Saving themodel
transfer_model.save('model/vggforsp500.h5')
