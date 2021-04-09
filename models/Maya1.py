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

# ### Directory preparation
# In case you have the train_shuffled_100cat.csv file downloaded in your environment, you can create the directories and images using the snippet below
# 
# *Important note: If you've downloaded the compressed directory from this notebook, skip the cell below*

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
#import matplotlib.pyplot as plt


# Register your gpu if you have one in the environment

# In[ ]:


#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices):
 #   tf.config.experimental.set_memory_growth(physical_devices[0], True)
#print(physical_devices)


# ## Define your models
# 
# ### 2. Define your  models, cost function, optimizer, learning rate
# Our convolutional model which gave the best result

# ### The pre-trained models

# #### The Inception pre-trained model using ImageNet 

# In[2]:


NUM_CATEGORIES=99
def get_inceptionv3_model(ishape = (180,180,3), k = 99, lr = 1e-4, train_base=False):
    input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
    base_model = tf.keras.applications.InceptionV3(
        input_shape=ishape,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=k,
        classifier_activation="softmax",
    )
    base_model.trainable = train_base
    x1 = base_model(input_layer, training=False)
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(k,activation = 'softmax')(x2)
    model = tf.keras.Model(inputs = input_layer, outputs =out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    print(model.summary())
    return model
inceptionv3_model = get_inceptionv3_model(train_base=False)


# In[3]:


seed = 909 # (IMPORTANT) to input image and corresponding target with same augmentation parameter.

gen_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',               "cval": 0}

train_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params) 

train_image_generator = train_image_datagen.flow_from_directory("../data-100cat/train/",
                                                    class_mode="categorical",  classes=[str(i) for i in range(99)], target_size=(180, 180), batch_size = 32,seed=seed,shuffle = True)

val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

val_image_generator = val_image_datagen.flow_from_directory("../data-100cat/val/",
                                                     class_mode="categorical",  classes=[str(i) for i in range(99)],batch_size = 32,seed=seed, target_size=(180, 180),color_mode='rgb',shuffle = True)

test_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

test_image_generator = val_image_datagen.flow_from_directory("../data-100cat/test/",
                                                     class_mode="categorical", classes=[str(i) for i in range(99)],batch_size = 32,seed=seed, target_size=(180, 180),color_mode='rgb')


# In[4]:


x, y = next(train_image_generator)
print("Train:", x.shape, y.shape, y[:10])
x, y = next(val_image_generator)
print("Val:", x.shape, y.shape, y[:10])
x, y = next(test_image_generator)
print("Test:", x.shape, y.shape, y[:10])


# Below we have a look at the generated data in all train/val/test set and verify that only the training set has been augmented.

# In[11]:


# x, y = next(train_image_generator)
# print(x.shape, y.shape)
# plt.figure(figsize = (6,4), dpi = 300)
# for ii in range(x.shape[0]):
#   plt.subplot(4,8,ii+1)
#   plt.imshow(x[ii])
#   plt.axis("off")
#   plt.title(y[ii])
# plt.show()
# xv, yv = next(val_image_generator)
# print(xv.shape, yv.shape)
# plt.figure(figsize = (6,4), dpi = 300)
# for ii in range(x.shape[0]):
#   plt.subplot(4,8,ii+1)
#   plt.imshow(xv[ii])
#   plt.axis("off")
#   plt.title(yv[ii])
# plt.show()


# ## 6. Define your callbacks (save your model, patience, etc.)

# In[5]:


from time import time; 
model_name_inceptionv3 = "cdiscount_beforetuning_minimumlayers_categorical_april6.h5".format(int(time()))

print("Saving model: {}".format(model_name_inceptionv3))
# define your callbacks
# remember that you need to save the weights of your best model!
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 6)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name_inceptionv3, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=True,                                             mode='min')
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%10 == 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)


# ## 7. Train your model

# In[10]:


k=99
input_layer = tf.keras.layers.Input(shape=(180,180,3), dtype="float")
base_model = tf.keras.applications.InceptionV3(
        input_shape=(180,180,3),
        include_top=False,
        weights="imagenet",
        input_tensor=input_layer,
        pooling=None,
        classes=k,
        classifier_activation="softmax",
    )
base_model.trainable = False
input_image = tf.keras.Input(shape=(180,180,3))
x1 = base_model.output
x2= tf.keras.layers.Flatten()(x1)
out = tf.keras.layers.Dense(k,activation = 'softmax')(x2)
model = tf.keras.Model(inputs=input_layer, outputs=out)
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


inceptionv3_model.fit(train_image_generator,batch_size=16, validation_data = (val_image_generator),                   steps_per_epoch=500, epochs=500,verbose=1, callbacks = [early_stop, monitor, lr_schedule])


# In[16]:


inceptionv3_model.load_weights(model_name_inceptionv3)
base_model.trainable = True
inceptionv3_model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


from time import time; 
model_name_tuned= "cdiscount_beforetuning_morelayers_categorical_aftertuning_moreepochs3.h5".format(int(time()))

print("Saving model: {}".format(model_name_tuned))
# define your callbacks
# remember that you need to save the weights of your best model!
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 6)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name_tuned, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=True,                                             mode='min')
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%10 == 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)


# In[2]:


inceptionv3_model.fit(train_image_generator,batch_size=16, validation_data = (val_image_generator),                    epochs=100,verbose=1, callbacks = [early_stop, monitor, lr_schedule])


# Testing the fine-tuning

# In[ ]:


model.load_weights(model_name_tuned)
metrics = model.evaluate(test_image_generator)


# 
