
import numpy as np
import pylab as plt
import pandas as pd

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam

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
MAIN EXECUTIONS

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
#y_test=np.array(Y_test_FutPredict_image)


