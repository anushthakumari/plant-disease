
#Extract zip files - plantvillage dataset
import zipfile
zip_ref = zipfile.ZipFile('./datasets/archive.zip','r')
zip_ref.extractall('./content')
zip_ref.close()

#import modules
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as py
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB3,preprocess_input
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img,img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras import regularizers
import numpy as np
from glob import glob
import pandas as p
import os
import cv2
from keras.models import load_model
import shutil
from sklearn.model_selection import train_test_split
# from PIL import image
from sklearn.metrics import accuracy_score


# Functions to Create Data Frame from Dataset
# Generate data paths with labels
def define_paths(img_dir):
    filepaths = [] #creating empty list
    labels = []

    folds = os.listdir(img_dir) # list directories inside img_dir
    for fold in folds:
        foldpath = os.path.join(img_dir, fold) #concatenate each dir inside img_dir with img_dir and storing them in foldpath
        filelist = os.listdir(foldpath) #now access image files
        for files_l in filelist:
            fpath = os.path.join(foldpath, files_l)
            filepaths.append(fpath) #append those image directories in filepaths
            labels.append(fold)# append folder name which are their corresponding label

    return filepaths, labels

# Concatenate data paths with labels into one dataframe ( to later be fitted into the model )
def define_df(files, classes):
    F = p.Series(files, name= 'file_dir') #creating series of file paths
    L = p.Series(classes, name='labels') #creating series of labels
    return p.concat([F, L], axis= 1)

# Split dataframe to train, valid, and test
def split_data(img_dir):
    # train dataframe
    files, classes = define_paths(img_dir) #calling define_paths function which will return file_paths and labels
    df = define_df(files, classes) #calling define_df function
    strat = df['labels'] #using stratify to uniformly distribute instances of classes in both dataframe , i.e. train and dummy to prevent model from getting biased towards any particular class.
    train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123, stratify= strat)

    # valid and test dataframe
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 123, stratify= strat)

    return train_df, valid_df, test_df


'''
    This function takes train, validation, and test dataframe and fit them into image data generator, because model takes data from image data generator.
    Image data generator converts images into tensors. '''
def create_gens (train_df, valid_df, test_df, batch_size):

    img_size = (300, 300) # setting it according to image size used in pretrained model
    channels = 3
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels) #defines that image has 300 X 300 pixels dimension and RGB system


    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))

    test_steps = ts_length // test_batch_size

    # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function= scalar, horizontal_flip= True)
    ts_gen = ImageDataGenerator(preprocessing_function= scalar)

    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'file_dir', y_col= 'labels', target_size= img_size, class_mode= 'sparse',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'file_dir', y_col= 'labels', target_size= img_size, class_mode= 'sparse',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'file_dir', y_col= 'labels', target_size= img_size, class_mode= 'sparse',
                                        color_mode= color, shuffle= False, batch_size= test_batch_size)

    return train_gen, valid_gen, test_gen



data_dir = './content/plantvillage dataset/segmented'


# Get splitted data
train_df, valid_df, test_df = split_data(data_dir)

# Get Generators
batch_size = 40
train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)


#Getting number of output classes
class_count = len(list(train_gen.class_indices.keys()))

#Defining pretrained model ,  Model used - EfficientNetB3
base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= (300,300,3), pooling= 'max')


#Using model's trained parameters without changing it
for layer in base_model.layers:
    layer.trainable = False



#Creating model
new_model = Sequential()
new_model.add(base_model)
new_model.add(Flatten())
new_model.add(BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001))
new_model.add(Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                activation= 'relu'))
new_model.add(Dropout(rate= 0.45, seed= 123))
new_model.add(Dense(class_count, activation= 'softmax'))

new_model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(0.0005),metrics=['accuracy'])
new_model.summary()

#Set callback parameter , saving best parameter in final_model_weights1.hdf5
checkpointer = [tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', verbose = 1, restore_best_weights=True, mode="max",patience = 10),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='final_model_weights1.hdf5',
                    monitor="val_accuracy",
                    verbose=1,
                    save_best_only=True,
                    mode="max")]




#implement earlystopping
steps_per_epoch = train_gen.n // train_gen.batch_size
validation_steps = valid_gen.n // valid_gen.batch_size
history = new_model.fit(x=train_gen,
                 validation_data=valid_gen,
                 epochs=25,
                 callbacks=[checkpointer],
                 steps_per_epoch=steps_per_epoch,
                 validation_steps=validation_steps,shuffle = False)



#plottting accuracy curve for training and validation data
py.plot(history.history['accuracy'],color='blue',label='train')
py.plot(history.history['val_accuracy'],color='red',label='validation')
py.legend()
py.show()


py.plot(history.history['loss'],color='blue',label='train')
py.plot(history.history['val_loss'],color='red',label='validation')
py.legend()
py.show()


#Evaluate model
test_batch  = 32;
train_eval = new_model.evaluate(train_gen,steps = test_batch,verbose=1) #checking for loss and accuracy for training data
val_eval = new_model.evaluate(valid_gen,steps = test_batch,verbose=1)#checking for loss and accuracy for validation data
test_eval = new_model.evaluate(test_gen,steps = test_batch,verbose=1)#checking for loss and accuracy for testing data
print("Train Loss: ", round(train_eval[0]*100,2))
print("Train Accuracy: ", round(train_eval[1]*100,2))
print('-' * 20)
print("Validation Loss: ", round(val_eval[0]*100,2))
print("Validation Accuracy: ", round(val_eval[1]*100,2))
print('-' * 20)
print("Test Loss: ", round(test_eval[0]*100,2))
print("Test Accuracy: ",round(test_eval[1]*100,2))

#Making Predictions
preds = new_model.predict(test_gen)

a = np.argmax(preds, axis=1)
test_labels = test_gen.classes

#checking accuracy score for test generators
accuracy_score(test_labels,a)


#Saving Model
new_model.save('./model/Plant_Detection_model_final.h5')
