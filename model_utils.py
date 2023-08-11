
from keras.utils import load_img,img_to_array
import numpy as np
from keras.models import load_model

#Loading saved model
model = load_model('./model/Plant_Detection_model_final.h5')


def predict_class(pred_number):
    class_name = "Empty"
    if(pred_number == 0): class_name = "Apple___Apple_scab"
    elif(pred_number == 1): class_name = "Apple___Black_rot"
    elif(pred_number == 2): class_name = "Apple___Cedar_apple_rust"
    elif(pred_number == 3): class_name = "Apple___healthy"
    elif(pred_number == 4): class_name = "Blueberry___healthy"
    elif(pred_number == 5): class_name = "Cherry_(including_sour)___Powdery_mildew"
    elif(pred_number == 6): class_name = "Cherry_(including_sour)___healthy"
    elif(pred_number == 7): class_name = "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    elif(pred_number == 8): class_name = "Corn_(maize)___Common_rust_"
    elif(pred_number == 9): class_name = "Corn_(maize)___Northern_Leaf_Blight"
    elif(pred_number == 10): class_name = "Corn_(maize)___healthy"
    elif(pred_number == 11): class_name = "Grape___Black_rot"
    elif(pred_number == 12): class_name = "Grape___Esca_(Black_Measles)"
    elif(pred_number == 13): class_name = "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
    elif(pred_number == 14): class_name = "Grape___healthy"
    elif(pred_number == 15): class_name = "Orange___Haunglongbing_(Citrus_greening)"
    elif(pred_number == 16): class_name = "Peach___Bacterial_spot"
    elif(pred_number == 17): class_name = "Peach___healthy"
    elif(pred_number == 18): class_name = "Pepper,_bell___Bacterial_spot"
    elif(pred_number == 19): class_name = "Pepper,_bell___healthy"
    elif(pred_number == 20): class_name = "Potato___Early_blight"
    elif(pred_number == 21): class_name = "Potato___Late_blight"
    elif(pred_number == 22): class_name = "Potato___healthy"
    elif(pred_number == 23): class_name = "Raspberry___healthy"
    elif(pred_number == 24): class_name = "Soybean___healthy"
    elif(pred_number == 25): class_name = "Squash___Powdery_mildew"
    elif(pred_number == 26): class_name = "Strawberry___Leaf_scorch"
    elif(pred_number == 27): class_name = "Strawberry___healthy"
    elif(pred_number == 28): class_name = "Tomato___Bacterial_spot"
    elif(pred_number == 29): class_name = "Tomato___Early_blight"
    elif(pred_number == 30): class_name = "Tomato___Late_blight"
    elif(pred_number == 31): class_name = "Tomato___Leaf_Mold"
    elif(pred_number == 32): class_name = "Tomato___Septoria_leaf_spot"
    elif(pred_number == 33): class_name = "Tomato___Spider_mites Two-spotted_spider_mite"
    elif(pred_number == 34): class_name = "Tomato___Target_Spot"
    elif(pred_number == 35): class_name = "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    elif(pred_number == 36): class_name = "Tomato___Tomato_mosaic_virus"
    elif(pred_number == 37): class_name = "Tomato___healthy"
    return class_name


def predict(image_path):
    img = load_img(image_path,target_size = (300,300))
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)
    pred = model.predict(x)
    p = np.argmax(pred)
    class_prediction = predict_class(p)
    return class_prediction

