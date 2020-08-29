import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

'''
Here we have a trained model model/vggforsp500.h5 and datas for testing 
datas/X_test_image.csv
datas/Y_test_StateClass_image.csv
datas/Y_test_FutPredict_image.csv

'''
trained_model_path='model/vggforsp500.h5'

##
'''
UTILITY FUNCTIONS
to put in another file
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
##

#recuperation of testing datas and organising it 
X_test_image=pd.read_csv('datas/X_test_image.csv')
Y_test_StateClass_image=pd.read_csv('datas/Y_test_StateClass_image.csv')
Y_test_FutPredict_image=pd.read_csv('datas/Y_test_FutPredict_image.csv')

#setting up the index to Date
X_test_image=X_test_image.set_index("Date")
Y_test_StateClass_image=Y_test_StateClass_image.set_index("Date")
Y_test_FutPredict_image=Y_test_FutPredict_image.set_index("Date")

#modify dataset to np array for input to NN
x_test=change_X_df__nparray_image(X_test_image)
y_test_state=np.array(Y_train_StateClass_image)
y_test_value=np.array(Y_train_FutPredict_image)

##Setting up xtest and ytest
#Here we focus on predicting the future state Y_train_StateClass_image
nb_test=len(X_test_image.index)
x_test=np.zeros((nb_test,32,32,3))
for i in range(nb_test):
  tmp=np.array(X_test_image.iloc[i])
  tmp=tmp.reshape(32,32,3)
  x_test[i]=tmp

y_test=np.array(Y_test_StateClass_image)
#y_test=np.array(Y_test_FutPredict_image)

#In our example we need to y into categorical as it has 6 categories
nb_classes=6
y_test = np_utils.to_categorical(y_test, nb_classes)

############
#recuperation of model
vggsp500model = load_model(trained_model_path)

#Evaluate the model on the test data
score  = vggsp500model.evaluate(x_test, y_test)


Y_pred = vggsp500model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
#y= np.argmax(y_test)
y=y_test
print('Confusion Matrix')

target_state = ['SS', 'SN', 'N','NB','BB']
def statetostring(x):
  return target_state[int(x)]
sY_pred=[statetostring(i) for i in y_pred]
sY_real=[statetostring(i) for i in y]

#matrice  de confusion
mat=confusion_matrix(sY_real, sY_pred, normalize='true', labels=target_state)
df_confmat=pd.DataFrame(mat,index=target_state, columns=target_state)

#Accuracy on test data
print('Accuracy on the Test Images: ', score[1])
#matrice  de confusion
print(df_confmat)

# Classification report
print('classification report')
print(classification_report(sY_real, sY_pred, target_names=target_state))
