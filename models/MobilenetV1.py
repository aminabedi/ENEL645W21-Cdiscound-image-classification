#!/usr/bin/env python
# coding: utf-8

# # Project - Cdiscount Image Classification
# 
# 

# ## Data ingestion
# ## 1. Intro
# The primary training set is a 57GB bson file, having ~15 Million images (180x180 images in Base64 format) of ~7.06 Million products. We have imported the dataset into a MongoDB instance on a VPS, so we were able to query among the records.
# We have chosen 100 categories, which overally consist of ~246K images of ~110K products.
# 

# #### Dataset preparation
# 
# First we need to ensure that the "gdown" library is installed and accessible in the environment and download the train_shuffled_100cat data from Google Drive:

# In[6]:


#! pip install gdown


# In[ ]:


# ! gdown --id 1JGaRoMrVAUregwwd_SpEJA-xjHRKMn9h
# ! tar -xvzf data-100cat.tar


# ### Directory preparation
# In case you have the train_shuffled_100cat.csv file downloaded in your environment, you can create the directories and images using the snippet below
# 
# *Important note: If you've downloaded the compressed directory from this notebook, skip the cell below*

# In[3]:


# import pandas as pd
# import base64
# import io
# from pathlib import Path

# FILE="train_shuffled_100cat.csv"

    

# df=pd.read_csv(FILE, header=3)
# df.describe()

# categories = df['category_id'].unique()
# categories.sort()
# category_id_map = {k: v for v, k in enumerate(categories)}
# df["class"] = df["category_id"].apply(lambda x: category_id_map[x])

# rdf = df.sample(frac=1, random_state=123)
# rdf.reset_index(drop=True, inplace=True)
# count = rdf.shape[0]
# num_train = int(count * .75) #= splitting point of train/val set
# num_val = num_train + int(count * .2)

# for idx, rec in rdf.iterrows():
#     folder = "train" if idx < num_train else ("val" if idx < num_val else "test")
#     classname = rec["class"]
#     Path("data-100cat/%s/%d"%(folder, classname)).mkdir(parents=True, exist_ok=True)
#     fh = open("data-100cat/%s/%d/%d-%d-%d.jpg"%(folder,  classname, rec["id"], idx, classname ) , "wb")
#     fh.write(
#                 base64.b64decode(
#                     rec["image"]
#                 )
#             )
#     fh.close()
#     if idx % 10000==0:
#         print(idx, "Done")
    


# #### Note for the team
# Since the original dataset is pretty large, I've created a subset file containing ~250K photos in 100 categories, exported the base64 images into train/val/test directories and in each directory, separated the samples into subdirectories named with their labels.

# ## Environment setup

# Import the required libraries

# In[1]:


import pandas as pd

import base64
from PIL import Image
import base64
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

NUM_CATEGORIES = 99
DATA_ROOT="data-100cat/"


# Register your gpu if you have one in the environment

# In[2]:


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices):
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# print(physical_devices)


# In[3]:


seed = 909 # (IMPORTANT) to input image and corresponding target with same augmentation parameter.

gen_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',               "cval": 0}

train_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params) 

train_image_generator = train_image_datagen.flow_from_directory(DATA_ROOT+"train/",
                                                    class_mode="categorical",  classes=[str(i) for i in range(99)], target_size=(180, 180), batch_size = 32,seed=seed,shuffle = True)

val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

val_image_generator = val_image_datagen.flow_from_directory(DATA_ROOT+"val/",
                                                     class_mode="categorical",  classes=[str(i) for i in range(99)],batch_size = 32,seed=seed, target_size=(180, 180),color_mode='rgb',shuffle = True)

test_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

test_image_generator = val_image_datagen.flow_from_directory(DATA_ROOT+"test/",
                                                     class_mode="categorical", classes=[str(i) for i in range(99)],batch_size = 32,seed=seed, target_size=(180, 180),color_mode='rgb')


# In[7]:


x, y = next(train_image_generator)
print("Train:", x.shape, y.shape, y[:10], y.max(), np.unique(y))
x, y = next(val_image_generator)
print("Val:", x.shape, y.shape, y[:10], y.max(), np.unique(y))
x, y = next(test_image_generator)
print("Test:", x.shape, y.shape, y[:10], y.max(), np.unique(y))


# In[8]:


# x, y = next(train_image_generator)
# print(x.shape, y.shape)
# plt.figure(figsize = (6,4), dpi = 300)
# for ii in range(x.shape[0]):
#   plt.subplot(4,8,ii+1)
#   plt.imshow(x[ii])
#   plt.axis("off")
#   plt.title(y[ii].argmax())
# plt.show()
# xv, yv = next(val_image_generator)
# print(xv.shape, yv.shape)
# plt.figure(figsize = (6,4), dpi = 300)
# for ii in range(x.shape[0]):
#   plt.subplot(4,8,ii+1)
#   plt.imshow(xv[ii])
#   plt.axis("off")
#   plt.title(yv[ii].argmax())
# plt.show()


# ## Models
# 
# ### Our trained-from-scratch convolutional model which gave the best result, but far less than the pre-trained ones

# In[4]:



def my_model_cnn(ishape = (180,180,3), k = NUM_CATEGORIES, lr = 1e-3):
    input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
    l1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation= 'relu')(input_layer)
    l2 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation= 'relu')(l1)
    l2_drop = tf.keras.layers.Dropout(0.25)(l2)
    l3 = tf.keras.layers.MaxPool2D((2,2))(l2_drop)
    l4 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation='relu')(l3)
    l5 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation='relu')(l4)
    l5_drop = tf.keras.layers.Dropout(0.25)(l5)
    l6 = tf.keras.layers.MaxPool2D((2,2))(l2_drop)
    l7 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same', activation='relu')(l6)
    l8 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same', activation='relu')(l7)
    l8_drop = tf.keras.layers.Dropout(0.25)(l8)
    l9 = tf.keras.layers.MaxPool2D((2,2))(l8_drop)
    l10 = tf.keras.layers.Conv2D(256, (3,3), padding = 'same', activation='relu')(l9)
    l11 = tf.keras.layers.Conv2D(256, (3,3), padding = 'same', activation='relu')(l10)
    l11_drop = tf.keras.layers.Dropout(0.25)(l11)
    l12 = tf.keras.layers.MaxPool2D((2,2))(l11_drop)
    l13 = tf.keras.layers.Conv2D(512, (3,3), padding = 'same', activation='relu')(l12)
    l14 = tf.keras.layers.Conv2D(512, (3,3), padding = 'same', activation='relu')(l13)
    l14_drop = tf.keras.layers.Dropout(0.25)(l14)
    flat = tf.keras.layers.Flatten()(l14_drop)
    out = tf.keras.layers.Dense(k, activation= 'softmax')(flat)
    model = tf.keras.models.Model(inputs = input_layer, outputs = out)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = 'sparse_categorical_crossentropy', metrics= ["accuracy"])
    return model
cnn_model = my_model_cnn()
print(cnn_model.summary())


# In[7]:


from time import time; 
cnn_model_name = "cdiscount_cnn_{}.h5".format(int(time()))#"cdiscount_1617041471.h5"#

print("Saving model: {}".format(cnn_model_name))
# define your callbacks
# remember that you need to save the weights of your best model!
cnn_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 6)

cnn_monitor = tf.keras.callbacks.ModelCheckpoint(cnn_model_name, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=True,                                             mode='min')
# Learning rate schedule
def cnn_scheduler(epoch, lr):
    if epoch%10 == 0:
        lr = lr/2
    return lr

cnn_lr_schedule = tf.keras.callbacks.LearningRateScheduler(cnn_scheduler,verbose = 0)


# In[ ]:


cnn_model.fit(train_image_generator, steps_per_epoch=1000, validation_data = (val_image_generator),                    validation_steps = 100,                    epochs=500, verbose=1, callbacks = [cnn_early_stop, cnn_monitor, cnn_lr_schedule])


# ### The pre-trained models

# #### The MobileNet pre-trained model using ImageNet weights

# In[8]:


ishape = (180,180,3)
input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
base_model = tf.keras.applications.MobileNet(
    input_shape=ishape,
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.01,
    include_top=False,
    weights="imagenet",
    input_tensor=input_layer,
    pooling=None,
    classes=NUM_CATEGORIES,
    classifier_activation="softmax",
)
base_model.trainable = False
x1 = base_model.output
x2 = tf.keras.layers.Flatten()(x1)
out = tf.keras.layers.Dense(NUM_CATEGORIES,activation = 'softmax')(x2)
mobilenet_model = tf.keras.Model(inputs = input_layer, outputs =out)
mobilenet_model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4),
          loss='categorical_crossentropy',
          metrics=['accuracy'])
print(mobilenet_model.summary())


# Below we have a look at the generated data in all train/val/test set and verify that only the training set has been augmented.

# ## 6. Specify the callbacks

# In[15]:


from time import time; 
model_name = "cdiscount_mobilenet{}.h5".format(int(time()))#

print("Saving model: {}".format(model_name))
# define your callbacks
# remember that you need to save the weights of your best model!
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 6)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=True,                                             mode='min')
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%10 == 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)


# ## 7. Train your model

# In[13]:


mobilenet_model.fit(train_image_generator, steps_per_epoch=1000, validation_data = (val_image_generator),                    validation_steps = 100,                    epochs=100, verbose=1, callbacks = [early_stop, monitor, lr_schedule])


# ### Fine-tune your pre-trained model

# In[21]:


mobilenet_model.load_weights(model_name)
base_model.trainable = True

mobilenet_model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-6),
          loss='categorical_crossentropy',
          metrics=['accuracy'])

print(mobilenet_model.summary())
mobilenet_model.fit(train_image_generator, batch_size = 32, epochs = 5,           verbose = 1, callbacks= [early_stop, monitor, lr_schedule], validation_data=(val_image_generator))


# ## 8. Test your model

# In[11]:


print(model_name)
mobilenet_model.load_weights(model_name)
metrics = mobilenet_model.evaluate(test_image_generator)

