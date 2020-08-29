
import numpy as np
import pylab as plt

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam


##First sequential model construction
def simple_moodel_create():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',input_shape = x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(6, activation='softmax'))
    return model

model=simple_model_create()   

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

##getting the datas
pd.read_csv('datas/df_table_image_tocsv.zip)

##Fitting the model on the train data and labels.
model.fit(x_train, y_train, batch_size=32, epochs=10, 
          verbose=1, validation_split=0.2, shuffle=True)

def main():
    print("end train")
main()
