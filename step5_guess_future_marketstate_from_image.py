import numpy as np
from PIL import Image

#path for trained model
trained_model_path='model/vggforsp500.h5'

#path for the image taken by the user
image_path='ImageM/image1.PNG'

#Load the image and resize it to 32x32 and taking off the transparency
load_img_rz = np.array(Image.open('image_path').resize((32,32)))
#Image.fromarray(load_img_rz).save('/content/drive/My Drive/_sample_data/ImageM/image1.PNG')
image=load_img_rz[:,:,:3]/255
print("After resizing:",load_img_rz.shape)

############
#recuperation of the model
vggsp500model = load_model(trained_model_path)
Y_pred = vggsp500model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)

target_state = ['SS', 'SN', 'N','NB','BB']
df_result=pd.DataFrame((Y_pred))

df_result.columns=target_state
df_result.index=[image_path, image_path+'1']
print(df_result)
print ("for ",image_path, "the best result is ", target_state[int(y_pred[0])] )
