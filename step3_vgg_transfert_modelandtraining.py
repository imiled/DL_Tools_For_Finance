
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

#Importing the VGG16 model
from keras.applications.vgg16 import VGG16, preprocess_input

#Importing the VGG16 model
from keras.applications.vgg16 import VGG16, preprocess_input

batch_size=32
epochs=50


#Loading the VGG16 model with pre-trained ImageNet weights
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg_model.trainable = False # remove if you want to retrain vgg weights

vgg_model.summary()

#At this stage we have x and y to train the model
#In our example we need to y into categorical as it has 6 categories
nb_classes=6
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

##Transfert model from vgg
transfer_model = Sequential()
transfer_model.add(vgg_model)
transfer_model.add(Flatten())
transfer_model.add(Dense(128, activation='relu'))
transfer_model.add(Dropout(0.2))
transfer_model.add(Dense(6, activation='softmax'))

##Display summary of neural network
transfer_model.summary()

transfer_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

##Fitting the model on the train data and labels.
##we choose here the stat to predict
model.fit(x_train, y_train_state, batch_size=32, epochs=10, 
          verbose=1, validation_split=0.2, shuffle=True)

history = transfer_model.fit(x_train, y_train, \
                              batch_size=batch_size, epochs=epochs, \
                              validation_split=0.2, verbose=1, shuffle=True)

# Saving themodel
transfer_model.save('model/vggforsp500.h5')

new_model = keras.models.load_model('path_to_my_model.h5')

#Evaluate the model on the test data
score  = new_model.evaluate(x_test, y_test)

#Accuracy on test data
print('Accuracy on the Test Images: ', score[1])

