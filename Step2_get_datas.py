import pandas as pd
import numpy as np

X_train_image.read_csv('datas/X_train_image.csv')
Y_train_StateClass_image.read_csv('datas/Y_train_StateClass_image.csv')
Y_train_FutPredict_image.read_csv('datas/Y_train_FutPredict_image.csv')

X_test_image.read_csv('datas/X_test_image.csv')
Y_test_StateClass_image.read_csv('datas/Y_test_StateClass_image.csv')
Y_test_FutPredict_image.read_csv('datas/Y_test_FutPredict_image.csv')

#modify dataset to np array for input to NN

x_test=change_X_df__nparray_image(X_train_image)
y_train_state=np.array(Y_train_StateClass_image)
y_train_value=np.array(Y_train_FutPredict_image)

x_test=change_X_df__nparray_image(X_test_image)
y_test_state=np.array(Y_train_StateClass_image)
y_test_value=np.array(Y_train_FutPredict_image)
