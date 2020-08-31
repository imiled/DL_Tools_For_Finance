
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

#path for trained model
trained_model_path='model/vggforsp500.h5'

#path for the image taken by the user
image_path='ImageM/image1.PNG'

#Load the image and resize it to 32x32 and taking off the transparency
load_img_rz = np.array(Image.open(image_path).resize((32,32)))
#Image.fromarray(load_img_rz).save('/content/drive/My Drive/_sample_data/ImageM/image1.PNG')
image=load_img_rz[:,:,:3]/255
print("After resizing:",image.shape)

#petite astuce pour ne pas avoir d erreur avec les types list, tensors,  nparray et dataframe
doubleimage=np.array([image,image])
############
#recuperation of the model
vggsp500model = load_model(trained_model_path)
Y_pred = vggsp500model.predict(doubleimage)
y_pred = np.argmax(Y_pred, axis=1)

target_state = ['SS', 'SN', 'N','NB','BB','Error']
df_result=pd.DataFrame((Y_pred))

df_result.columns=target_state
df_result.index=[image_path, image_path+'1']
print ("for ",image_path, "the best result is ", target_state[int(y_pred[0])] )
print(df_result)
