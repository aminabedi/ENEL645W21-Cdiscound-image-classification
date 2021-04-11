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

# First we need to ensure that the "gdown" library is installed and accessible in the environment and download the train_shuffled_100cat data from Google Drive:

# In[6]:


get_ipython().system(' pip install gdown')


# ### Downoad the dataset
# #### method 1: Download the directory directly and unzip it using the cell below

# In[ ]:


get_ipython().system(' gdown --id 1JGaRoMrVAUregwwd_SpEJA-xjHRKMn9h')
get_ipython().system(' tar -xvzf data-100cat.tar')


# #### method 2: Download the csv file of the dataset and create the directory using the two cells below

# In[ ]:


get_ipython().system(' gdown --id 1F6Xf4yiYxeFEN6qhrL3YBNs0Vhx0bXJ1')


# ### Directory preparation
# In case you have the train_shuffled_100cat.csv file downloaded in your environment, you can create the directories and images using the snippet below
# 
# *Important note: If you've downloaded the compressed directory from this notebook, skip the cell below*

# In[ ]:


import pandas as pd
import base64
import io
from pathlib import Path

FILE="train_shuffled_100cat.csv"

    

df=pd.read_csv(FILE, header=3)
df.describe()

categories = df['category_id'].unique()
categories.sort()
category_id_map = {k: v for v, k in enumerate(categories)}
df["class"] = df["category_id"].apply(lambda x: category_id_map[x])

rdf = df.sample(frac=1, random_state=123)
rdf.reset_index(drop=True, inplace=True)
count = rdf.shape[0]
num_train = int(count * .75) #= splitting point of train/val set
num_val = num_train + int(count * .2)

for idx, rec in rdf.iterrows():
    folder = "train" if idx < num_train else ("val" if idx < num_val else "test")
    classname = rec["class"]
    Path("data-100cat/%s/%d"%(folder, classname)).mkdir(parents=True, exist_ok=True)
    fh = open("data-100cat/%s/%d/%d-%d-%d.jpg"%(folder,  classname, rec["id"], idx, classname ) , "wb")
    fh.write(
                base64.b64decode(
                    rec["image"]
                )
            )
    fh.close()
    if idx % 10000==0:
        print(idx, "Done")
    


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

PLOT_ENABLED = False # Set to False in case of running on a cluster 

NUM_CATEGORIES = 99
DATA_ROOT="data-100cat/"


# Register your gpu if you have one in the environment

# In[2]:


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)


# ### Data generators
# 
# Below we have defined the data generators which will normalize the pixel values and also does the data augmentations using rotations and flips

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


# Below we take a look at the shapes of our train/val/test sets and the target values

# In[4]:


x, y = next(train_image_generator)
print("Train:", x.shape, y.shape, y[:1], y.max(), np.unique(y))
x, y = next(val_image_generator)
print("Val:", x.shape, y.shape, y[:1], y.max(), np.unique(y))
x, y = next(test_image_generator)
print("Test:", x.shape, y.shape, y[:1], y.max(), np.unique(y))


# Below we take a look at some of the images from training and validation set, as expected, only the training set has been augmented with variants of the images

# In[5]:


if PLOT_ENABLED:
    x, y = next(train_image_generator)
    print(x.shape, y.shape)
    plt.figure(figsize = (6,4), dpi = 300)
    for ii in range(x.shape[0]):
      plt.subplot(4,8,ii+1)
      plt.imshow(x[ii])
      plt.axis("off")
      plt.title(y[ii].argmax())
    plt.show()
    xv, yv = next(val_image_generator)
    print(xv.shape, yv.shape)
    plt.figure(figsize = (6,4), dpi = 300)
    for ii in range(x.shape[0]):
      plt.subplot(4,8,ii+1)
      plt.imshow(xv[ii])
      plt.axis("off")
      plt.title(yv[ii].argmax())
    plt.show()


# ## Models
# 
# ### Our trained-from-scratch convolutional model which gave the best result, but far less than the pre-trained ones

# In[6]:


def get_cnn_model(ishape = (180,180,3), lr = 1e-3):
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
    out = tf.keras.layers.Dense(NUM_CATEGORIES, activation= 'softmax')(flat)
    model = tf.keras.models.Model(inputs = input_layer, outputs = out)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = 'categorical_crossentropy', metrics= ["accuracy"])
    return model


# In[7]:


def get_mobilenet_model(ishape = (180,180,3), lr = 1e-4, dr = 0.1):
    input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
    base_model = tf.keras.applications.MobileNet(
        input_shape=ishape,
        dropout=dr,
        include_top=False,
        weights="imagenet",
        input_tensor=input_layer,
        classes=NUM_CATEGORIES,
        classifier_activation="softmax",
    )
    base_model.trainable = False
    x1 = base_model.output
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(NUM_CATEGORIES,activation = 'softmax')(x2)
    mobilenet_model = tf.keras.Model(inputs = input_layer, outputs =out)

    return mobilenet_model, base_model


# In[8]:


def get_vgg16_model(ishape = (180,180,3), lr = 1e-4, dr = 0.1):
    input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        input_shape=ishape,
        include_top=False,
        input_tensor=input_layer,
        classifier_activation="softmax",
    ) 
    base_model.trainable = False
    x1 = base_model.output
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(NUM_CATEGORIES,activation = 'softmax')(x2)
    model = tf.keras.Model(inputs = input_layer, outputs =out)
    return model, base_model


# In[9]:


def get_resnet_model(ishape = (180,180,3), lr = 1e-4):
    input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
    base_model = tf.keras.applications.ResNet152V2(
       input_shape=ishape,
       include_top=False,
       weights="imagenet",
       input_tensor=input_layer,
       classes=NUM_CATEGORIES,
       classifier_activation="softmax",
    )
    base_model.trainable = False
    x1 = base_model.output
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(NUM_CATEGORIES,activation = 'softmax')(x2)
    model = tf.keras.Model(inputs = input_layer, outputs =out)

    return model, base_model


# In[10]:


def get_inception_model(ishape = (180,180,3), lr = 1e-4):
    input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
    base_model = tf.keras.applications.InceptionV3(
        input_shape=ishape,
        include_top=False,
        weights="imagenet",
        input_tensor=input_layer,
    )
    base_model.trainable = False
    x1 = base_model.output
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(NUM_CATEGORIES,activation = 'softmax')(x2)
    model = tf.keras.Model(inputs = input_layer, outputs =out)
    return model, base_model


# In[11]:


mobilenet_model, mobilenet_base_model = get_mobilenet_model()
vgg16_model, vgg16_base_model = get_vgg16_model()
resnet_model, resnet_base_model = get_resnet_model()
inception_model, inception_base_model = get_inception_model()
models = {
    "cnn": {
        "model": get_cnn_model(),
        "base": None
    },
    "mobilenet": {
        "model": mobilenet_model,
        "base": mobilenet_base_model
    },
    "vgg16": {
        "model": vgg16_model,
        "base": vgg16_base_model
    },
    "resnet": {
        "model": resnet_model,
        "base": resnet_base_model
    },
    "inception": {
        "model": inception_model,
        "base": inception_base_model,
    }
}


# In[12]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 6)

# Learning rate schedule
def lr_scheduler(epoch, lr):
    if epoch%10 == 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose = 0)

TRAIN_EPOCHS = 20
TUNE_EPOCHS = 5
STEPS_PER_EPOCH = 500


# In[ ]:


for name in models:
    model, base = models[name]["model"], models[name]["base"]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filename = "cdiscount_{}.h5".format(name)
    try:
        model.load_weights(filename)
    except OSError as e:
        print("Weights for model <<{}>> has not been found, trainig from scratch.".format(name))
    monitor = tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_loss',                                             verbose=0, save_best_only=True,                                             save_weights_only=True,                                             mode='min')
    model.fit(train_image_generator, steps_per_epoch=STEPS_PER_EPOCH, validation_data = (val_image_generator),                    validation_steps = 100,                    epochs=TRAIN_EPOCHS, verbose=1, callbacks = [early_stop, monitor, lr_schedule])
    model.load_weights(filename)
    
    if base:
        base.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        model.fit(train_image_generator, steps_per_epoch=STEPS_PER_EPOCH, validation_data = (val_image_generator),                    validation_steps = 100,                    epochs=TUNE_EPOCHS, verbose=1, callbacks = [early_stop, monitor, lr_schedule])
        model.load_weights(filename)

    metrics = model.evaluate(test_image_generator)
    models[name]["metrics"] = metrics
    print("Model {} got metrics:".format(name), metrics)


# In[ ]:


for name in models:
    print("Model {} results on the test set:", models[name]["metrics"])

