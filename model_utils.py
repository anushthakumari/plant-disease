
from keras.utils import load_img,img_to_array
import numpy as np
from keras.models import load_model
from skimage import io
import keras.utils as image

model = load_model('./model.h5')

def predict(image_path):
   
    img = image.load_img(image_path, grayscale=False, target_size=(64, 64))
    show_img=image.load_img(image_path, grayscale=False, target_size=(200, 200))
    disease_class = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    #x = np.array(x, 'float32')
    x /= 255

    custom = model.predict(x)
    a=custom[0]
    ind=np.argmax(a)     
    return disease_class[ind]

