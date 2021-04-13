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


#get_ipython().system(' pip install gdown')


# ## Downoad the dataset
# #### method 1: Download the directory directly and unzip it using the cell below

# In[ ]:


#get_ipython().system(' gdown --id 1JGaRoMrVAUregwwd_SpEJA-xjHRKMn9h')
#get_ipython().system(' tar -xvzf data-100cat.tar')


# #### method 2: Download the csv file of the dataset and create the directory using the two cells below

# In[ ]:


#get_ipython().system(' gdown --id 1F6Xf4yiYxeFEN6qhrL3YBNs0Vhx0bXJ1')


# ## Download our models
# As an evaluator, you can easily download the h5 files for our models by running the cell below, this way (as intentionally set as default behavior of the notebook), the training cell would only load the weigths and skips the training part.
# In case you want to retrain a model, you only need to set the "skip" property of that model to False before running the training cell.

# In[ ]:


#get_ipython().system(' gdown --id 1JGaRoMrVAUregwwd_SpEJA-xjHRKMn9h')
#get_ipython().system(' tar -xvzf weights.tar')
#get_ipython().system(' mv weights/* .')


# ### Directory preparation
# In case you have the train_shuffled_100cat.csv file downloaded in your environment, you can create the directories and images using the snippet below
# 
# *Important note: If you've downloaded the compressed directory from this notebook, skip the cell below*

# In[ ]:


#import pandas as pd
#import base64
#import io
#from pathlib import Path

#FILE="train_shuffled_100cat.csv"

    

#df=pd.read_csv(FILE, header=3)
#df.describe()

#categories = df['category_id'].unique()
#categories.sort()
#category_id_map = {k: v for v, k in enumerate(categories)}
#df["class"] = df["category_id"].apply(lambda x: category_id_map[x])

#rdf = df.sample(frac=1, random_state=123)
#rdf.reset_index(drop=True, inplace=True)
#count = rdf.shape[0]
#num_train = int(count * .75) #= splitting point of train/val set
#num_val = num_train + int(count * .2)

#for idx, rec in rdf.iterrows():
#    folder = "train" if idx < num_train else ("val" if idx < num_val else "test")
#    classname = rec["class"]
#    Path("data-100cat/%s/%d"%(folder, classname)).mkdir(parents=True, exist_ok=True)
#    fh = open("data-100cat/%s/%d/%d-%d-%d.jpg"%(folder,  classname, rec["id"], idx, classname ) , "wb")
#    fh.write(
#                base64.b64decode(
#                    rec["image"]
#                )
#            )
#    fh.close()
#    if idx % 10000==0:
#        print(idx, "Done")
    


# ## Environment setup

# Import the required libraries

# In[2]:


import pandas as pd

import base64
from PIL import Image
import base64
import io
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

PLOT_ENABLED = False # Set to False in case of running on a cluster 

NUM_CATEGORIES = 99
DATA_ROOT="../data-100cat/"


# Register your gpu if you have one in the environment

# In[3]:


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)


# ### Data generators
# 
# Below we have defined the data generators which will normalize the pixel values and also does the data augmentations using rotations and flips

# In[4]:


seed = 909 # (IMPORTANT) to input image and corresponding target with same augmentation parameter.

gen_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',               "cval": 0}

train_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params) 

train_image_generator = train_image_datagen.flow_from_directory(DATA_ROOT+"train/",
                                                    class_mode="categorical",  classes=[str(i) for i in range(99)], target_size=(180, 180), batch_size = 16,seed=seed,shuffle = True)

train_image_datagen_non_augmented = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

train_image_generator_non_augmented = train_image_datagen_non_augmented.flow_from_directory(DATA_ROOT+"train/",
                                                    class_mode="categorical",  classes=[str(i) for i in range(99)], target_size=(180, 180), batch_size = 16,seed=seed,shuffle = True)


val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

val_image_generator = val_image_datagen.flow_from_directory(DATA_ROOT+"val/",
                                                     class_mode="categorical",  classes=[str(i) for i in range(99)],batch_size = 16,seed=seed, target_size=(180, 180),color_mode='rgb',shuffle = True)

test_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

test_image_generator = val_image_datagen.flow_from_directory(DATA_ROOT+"test/",
                                                     class_mode="categorical", classes=[str(i) for i in range(99)],batch_size = 16,seed=seed, target_size=(180, 180),color_mode='rgb')


# Below we take a look at the shapes of our train/val/test sets and the target values

# In[5]:


x, y = next(train_image_generator)
print("Train:", x.shape, y.shape, y[:1], y.max(), np.unique(y))
x, y = next(val_image_generator)
print("Val:", x.shape, y.shape, y[:1], y.max(), np.unique(y))
x, y = next(test_image_generator)
print("Test:", x.shape, y.shape, y[:1], y.max(), np.unique(y))


# Below we take a look at some of the images from training and validation set, as expected, only the training set has been augmented with variants of the images

# In[6]:


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


def get_effiecientnet_model(ishape = (180,180,3), lr = 1e-4):
    input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
    base_model = tf.keras.applications.EfficientNetB5(
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


# In[12]:


def get_vgg16_nonaug_model(ishape = (180,180,3), lr = 1e-4, dr = 0.1):
    input_layer = tf.keras.layers.Input(shape=ishape, dtype="float")
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        input_shape=ishape,
        include_top=False,
        input_tensor=input_layer,
        classifier_activation="softmax",
    ) 
    base_model.trainable = False
    x1 = base_model(input_layer, training=False)
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(NUM_CATEGORIES,activation = 'softmax')(x2)
    model = tf.keras.Model(inputs = input_layer, outputs =out)
    return model, base_model


# In[13]:


mobilenet_model, mobilenet_base_model = get_mobilenet_model()
vgg16_model, vgg16_base_model = get_vgg16_model()
resnet_model, resnet_base_model = get_resnet_model()
inception_model, inception_base_model = get_inception_model()
effiecientnet_model, effiecientnet_base_model = get_effiecientnet_model()
vgg16_nonaug_model, vgg16_nonaug_base_model = get_vgg16_nonaug_model()
models = {
    "cnn": {
        "model": get_cnn_model(),
        "base": None,
        "skip": True,
        "use_augmentation": True
    },
    "mobilenet": {
        "model": mobilenet_model,
        "base": mobilenet_base_model,
        "skip": True,
        "use_augmentation": True
    },
    "vgg16": {
        "model": vgg16_model,
        "base": vgg16_base_model,
        "skip": True,
        "use_augmentation": True
    },

    "inception": {
        "model": inception_model,
        "base": inception_base_model,
        "skip": False,
        "use_augmentation": True
    },
    "effiecientnet": {
        "model": effiecientnet_model,
        "base": effiecientnet_base_model,
        "skip": True,
        "use_augmentation": True
    },
    "vgg16_non_augmented": {
        "model": vgg16_nonaug_model,
        "base": vgg16_nonaug_base_model,
        "skip": False,
        "use_augmentation": False
    },
    "resnet": {
        "model": resnet_model,
        "base": resnet_base_model,
        "skip": True,
        "use_augmentation": True
    },
}


# Below we have defined some auxiliary methods to handle saving and restoring results.

# In[14]:


import json
def load_previous_results():
    try:
        with open('results.json') as file:
            results = json.load(file)
            print("Previous results is successfully loaded.\n", results)
            for i in results:
                for j in results[i]:
                    models[i][j] = results[i][j]    
    except OSError:
        print("Results file was not found, run the trainin cell to obtain results")
        
def without_keys(d, keys):
    return {x: {t: d[x][t] for t in d[x] if t not in keys} for x in d }
        
def save_results(results = models):

    with open('results.json', 'w') as file:
        results = without_keys(models, ["model", "base", "ignore"])
        json.dump(results, file)


# The latest results from previous runs is saved in results.json file after each phase of training, to load them instead of retraining all moedls, run the load_previous_results.

# In[15]:


load_previous_results()


# In[16]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 6)

# Learning rate schedule
def lr_scheduler(epoch, lr):
    if epoch%10 == 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose = 0)

TRAIN_EPOCHS = 8
TUNE_EPOCHS = 10
STEPS_PER_EPOCH = 500


# In[ ]:


for name in models:
    model, base, skip, aug = models[name]["model"], models[name]["base"], models[name]["skip"], models[name]["use_augmentation"]
    print("Compiling {} model".format(name))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filename = "cdiscount_{}.h5".format(name)
    try:
        print("Loading weights from {}".format(filename))
        model.load_weights(filename)
        models[name]["model"].load_weights(filename)
        print("Weights for model <<{}>> has been loaded".format(name))
    except OSError as e:
        print("Weights for model <<{}>> has not been found, trainig from scratch.".format(name))
    if skip:
        continue
    print("Training {} model".format(name))
    monitor = tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_loss',                                             verbose=0, save_best_only=True,                                             save_weights_only=True,                                             mode='min')
    
    model.fit(train_image_generator if aug else train_image_generator_non_augmented ,
#               steps_per_epoch=STEPS_PER_EPOCH, 
              validation_data = (val_image_generator),\
#                     validation_steps = 100,\
                    epochs=TRAIN_EPOCHS, verbose=1, callbacks = [early_stop, monitor, lr_schedule])
    model.load_weights(filename)
    models[name]["before_tuning"] = model.evaluate(test_image_generator)
    print("Model {} got metrics, before tuning:".format(name), models[name]["before_tuning"])
    if base:
        print("Fine tuning {} model".format(name))
        base.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        model.fit(train_image_generator if aug else train_image_generator_non_augmented , 
#                   steps_per_epoch=STEPS_PER_EPOCH, 
                  validation_data = (val_image_generator),\
#                     validation_steps = 100,\
                    epochs=TUNE_EPOCHS, verbose=1, callbacks = [early_stop, monitor, lr_schedule])
        model.load_weights(filename)

    models[name]["metrics"] = model.evaluate(test_image_generator)
    print("Model {} got metrics, after tuning:".format(name), models[name]["metrics"])


# In[19]:


for name in models:
    try:
        print("Model {} results on the test set before fine tuning:".format(name),
              models[name]["before_tuning"],
              "and {} for final".format(models[name]["metrics"]))
    except Exception as e:
        print(e)
        print("Model {} results are not stored")


# In[20]:


save_results()


# In[21]:


x_samples, y_samples= [], []
while len(x_samples) < 20:
    x, y = next(test_image_generator)
    x_samples.append(x)
    y_samples.append(y)
for name in models:
    batch_idx = 0
    model = models[name]["model"]
    wrongs = []
    print("False predictions by {}:".format(name))
    plt.figure(figsize = (10,7), dpi = 300)
    while len(wrongs)< 10 and batch_idx<20:
        x, y = x_samples[batch_idx], y_samples[batch_idx]
        batch_idx +=1
        preds = models[name]["model"].predict(x)
        for ii in range(x.shape[0]):
            if preds[ii].argmax() != y[ii].argmax():
                wrongs.append({"img": x[ii, :, :, :], "label": y[ii].argmax(), "pred": preds[ii].argmax()})
        print("Found {} wrong predictions for {} model in current batch.".format(len(wrongs), name))
    for idx in range(10):
        item = wrongs[idx]
        plt.subplot(2, 5, idx + 1)
        plt.imshow(item["img"])
        plt.axis("off")
        plt.title("L: {}, P:{}".format(item["label"], item["pred"]))
    plt.show()


# ### Conclusion
# Below, for a sample of 40 images, we show how many of our models have failed to provide the correct prediction,
# For those which most of the models failed to provide, it is safe to say that the image is an outlier or the target class did not have a good quality in the images. For those which most of the models have provided correct predictions, we could easily use an ensemble of the models to cover them!

# In[23]:


wrongs, batch_idx = [], 0
plt.figure(figsize = (10,7), dpi = 300)
while len(wrongs)< 20 and batch_idx<20:
    x, y = x_samples[batch_idx], y_samples[batch_idx]
    batch_idx +=1
    preds = {}
    for name in models:
        preds[name] = models[name]["model"].predict(x)
    for ii in range(x.shape[0]):
        n_wrongs = 0
        for name in models:
            if preds[name][ii].argmax() != y[ii].argmax():
                n_wrongs += 1
        if n_wrongs:
            wrongs.append({"img": x[ii, :, :, :], "label": y[ii].argmax(), "n_wrongs": n_wrongs})
for idx in range(20):
    item = wrongs[idx]
    plt.subplot(4, 5, idx + 1)
    plt.imshow(item["img"])
    plt.axis("off")
    plt.title("L: {}, N:{}/7".format(item["label"], item["n_wrongs"]))
plt.show()

