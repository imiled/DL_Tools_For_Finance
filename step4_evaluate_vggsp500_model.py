import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

#########
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

############
#recuperation of model
vggsp500model = load_model('model/vggforsp500.h5')

#Evaluate the model on the test data
score  = vggsp500model.evaluate(x_test, y_test)

#Accuracy on test data
print('Accuracy on the Test Images: ', score[1])

Y_pred = vggsp500model.predict_generator(x_test)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Cats', 'Dogs', 'Horse']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
