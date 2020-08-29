
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


##First sequential model construction
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape = x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())
model.add(Dense(6, activation='softmax'))

##Display summary
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

##Fitting the model on the train data and labels.
##we choose here the stat to predict
model.fit(x_train, y_train_state, batch_size=32, epochs=10, 
          verbose=1, validation_split=0.2, shuffle=True)

