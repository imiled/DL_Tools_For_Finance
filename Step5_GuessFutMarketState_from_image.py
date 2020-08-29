import numpy as np
from PIL import Image

load_img_rz = np.array(Image.open('/content/drive/My Drive/_sample_data/ImageM/image1.PNG').resize((32,32)))
#Image.fromarray(load_img_rz).save('/content/drive/My Drive/_sample_data/ImageM/image1.PNG')
image=load_img_rz[:,:,:3]
print("After resizing:",load_img_rz.shape)
