# Databricks notebook source
# MAGIC %md
# MAGIC <h2 style="color:rgb(0,120,170)"> Predictive Analysis - Image Processing</h2>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Classify handwritten digits using the famous MNIST data
# MAGIC 
# MAGIC The goal of this task is to take an image of a handwritten single digit and determine what digit it is.  
# MAGIC 
# MAGIC The data for this competition was taken from the **MNIST data set**. The MNIST ("Modified National Institute of Standards and Technology") data set is a classic within the Machine Learning community that has been extensively studied.  More details about the dataset, including Machine Learning algorithms that have been tried on it and their levels of success, can be found [here](http://yann.lecun.com/exdb/mnist/index.html).

# COMMAND ----------

import os
import sys
import time
import glob
import math
import random
import pickle
import pylab
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot

from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.decomposition import PCA

import tensorflow as tf

%matplotlib inline
#matplotlib.rcdefaults()
#matplotlib.verbose.set_level('silent')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from IPython.display import Image
from IPython.core.display import HTML

# COMMAND ----------

datapath = os.path.join(os.getcwd(), 'data')
print(datapath)

# COMMAND ----------

# read training data from CSV file 

with ZipFile(os.path.join(datapath, 'kaggle_digits_train.zip'), 'r') as myzip:
    with myzip.open('kaggle_digits_train.csv') as myfile:
        train_data = pd.read_csv(myfile)
        
with ZipFile(os.path.join(datapath, 'kaggle_digits_test.zip'), 'r') as myzip:
    with myzip.open('kaggle_digits_test.csv') as myfile:
        test_data = pd.read_csv(myfile)

print('data({0[0]},{0[1]})'.format(train_data.shape))
train_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Every image is a "stretched" array of pixel values.
# MAGIC In this case it's  28 * 28 = 784 pixels.

# COMMAND ----------

sample_size = 4200

images = train_data.iloc[:sample_size, 1:].values   # Decreasing the number of images to save memory
images = images.astype(np.float16)

# convert from [0:255] to [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images numpy array have shape: ({0[0]},{0[1]})'.format(images.shape))

# COMMAND ----------

images

# COMMAND ----------

image_size = images.shape[1]
print ('image_size   => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print ('image_width  => {0}\nimage_height => {1}'.format(image_width,image_height))

# COMMAND ----------

# MAGIC %md
# MAGIC Let us take a look at the images in a way that is more comprehensible to the human eye. 

# COMMAND ----------

IMAGE_TO_DISPLAY = 10
#plt.install_repl_displayhook()
#plt.imshow(images[IMAGE_TO_DISPLAY].reshape((28, 28)), cmap=cm.binary)

# COMMAND ----------

labels_flat = train_data.iloc[:sample_size, 0].values   # Adjusting the number of labels to the number of images selected

print('labels_flat({0})'.format(len(labels_flat)))
print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))

# COMMAND ----------

plt.figure(figsize=(12,6))
for i in range(0,9):
    plt.subplot(250 + (i+1))
    img = images[i,:].reshape(28, 28)
    plt.imshow(img, cmap='Greys')
    plt.title(labels_flat[i])

# COMMAND ----------

labels_count = np.unique(labels_flat).shape[0]

print('labels_count => {0}'.format(labels_count))

# COMMAND ----------

test_data.head()

# COMMAND ----------

test_images = test_data.values.astype(np.float16)
test_images.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Just as we did for all the previously discussed models in the last lessons, we split our data into train and test sets. 

# COMMAND ----------

X_train, X_test, y_train, y_test = model_selection.train_test_split(images, 
                                                                    labels_flat, 
                                                                    test_size=0.4, 
                                                                    random_state=0)
print('train images({0[0]},{0[1]})'.format(X_train.shape))
print('validation images({0[0]},{0[1]})'.format(X_test.shape))
print('train labels({})'.format(y_train.shape))
print('validation labels({})'.format(y_test.shape))

# COMMAND ----------

del(images)  #save memory

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
clf_rf = RandomForestClassifier(n_estimators=300, 
                                criterion='gini', 
                                max_depth=None, 
                                min_samples_split=3, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='auto', 
                                max_leaf_nodes=None, 
                                bootstrap=True, 
                                oob_score=False, 
                                n_jobs=-1, 
                                random_state=0, 
                                verbose=0, 
                                warm_start=False, 
                                class_weight=None).fit(X_train, y_train)

eval_rf = clf_rf.score(X_test, y_test)
print(eval_rf)

# COMMAND ----------

# Train SVM...
from sklearn import svm
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

clf_svm = svm.SVC(kernel='poly',
                  C=1.57,
                  degree=2, 
                  gamma=0.278,
                  coef0=0.0, 
                  shrinking=True, 
                  probability=False, 
                  tol=0.001, 
                  cache_size=200, 
                  class_weight=None, 
                  verbose=False, 
                  max_iter=-1, 
                  random_state=0).fit(X_train, y_train)

eval_svm = clf_svm.score(X_test, y_test)
print(eval_svm)

# COMMAND ----------

# MAGIC %md
# MAGIC Making Predictions

# COMMAND ----------

predict_rf = clf_rf.predict(test_images)
predict_rf[0:10]

# COMMAND ----------

predict_svm = clf_svm.predict(test_images)
predict_svm[0:10]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Tensor Flow (with a fully connected ANN)

# COMMAND ----------

import tensorflow as tf

# COMMAND ----------

X_train = X_train.reshape(X_train.shape[0],28,28)
X_test = X_test.reshape(X_test.shape[0],28,28)
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

# COMMAND ----------

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation='softmax')
                                   ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# COMMAND ----------

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10, verbose=1)

# COMMAND ----------

model.evaluate(X_test, y_test, verbose=1)

# COMMAND ----------

# list all data in history
print(history.history.keys())

# COMMAND ----------

# summarize history for accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# COMMAND ----------

# summarize history for loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Tensor Flow (with a CNN)

# COMMAND ----------

# MAGIC %md
# MAGIC ### a) Simple model

# COMMAND ----------

# reshape dataset to have a single channel
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
#y_train = y_train.reshape(y_train.shape[0])
#y_test = y_test.reshape(y_test.shape[0])

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, 
                                                           kernel_size=(3, 3), 
                                                           activation='relu', 
                                                           kernel_initializer='he_uniform', 
                                                           input_shape=(28, 28, 1)
                                                          ),
                                    tf.keras.layers.MaxPooling2D((2, 2)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
                                    tf.keras.layers.Dense(10, activation='softmax'),
                                   ])

model.compile(optimizer='adam', #SGD(lr=0.01, momentum=0.9)
              loss="sparse_categorical_crossentropy",
              #loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# COMMAND ----------

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10, verbose=1)

# COMMAND ----------

model.evaluate(X_test, y_test, verbose=1)

# COMMAND ----------

# summarize history for accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# COMMAND ----------

# summarize history for loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### b) Slightly bigger model

# COMMAND ----------

from tensorflow.keras.callbacks import EarlyStopping

# COMMAND ----------

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, 
                                 kernel_size=(3, 3),
                                 activation='relu',
                                 kernel_initializer='he_uniform',
                                 input_shape=(28, 28, 1)
                                ))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', #tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
              #loss='categorical_crossentropy',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.summary()

# COMMAND ----------

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping])

# COMMAND ----------

model.evaluate(X_test, y_test, verbose=1)

# COMMAND ----------

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

hist_df = pd.DataFrame(history.history)
hist_df.head()

# COMMAND ----------

fig = plt.figure(figsize=(14,6))
plt.style.use('bmh')
params_dict = dict(linestyle='solid', linewidth=0.25, marker='o', markersize=6)

plt.subplot(121)
plt.plot(hist_df.loss, label='Training loss', **params_dict)
plt.plot(hist_df.val_loss, label='Validation loss', **params_dict)
plt.title('Loss for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(hist_df.accuracy, label='Training accuracy', **params_dict)
plt.plot(hist_df.val_accuracy, label='Validation accuracy', **params_dict)
plt.title('Accuracy for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ### c) [Using the LENET architecture](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/  )
# MAGIC 
# MAGIC ![](./img/lenet_architecture-768x226.png)
# MAGIC 
# MAGIC The LeNet architecture was first introduced by LeCun et al. in their 1998 paper, Gradient-Based Learning Applied to Document Recognition. As the name of the paper suggests, the authors’ implementation of LeNet was used primarily for OCR and character recognition in documents.  
# MAGIC 
# MAGIC The LeNet architecture is straightforward and small, (in terms of memory footprint), making it perfect for teaching the basics of CNNs — it can even run on the CPU (if your system does not have a suitable GPU), making it a great “first CNN”.  
# MAGIC 
# MAGIC However, if you do have GPU support and can access your GPU via Keras, you will enjoy extremely fast training times (in the order of 3-10 seconds per epoch, depending on your GPU).

# COMMAND ----------

model = tf.keras.Sequential()
model.add(tf.keras.layers.Convolution2D(filters=20, kernel_size = (5, 5), padding = "same", input_shape = (28, 28, 1)))
model.add(tf.keras.layers.Activation(activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides =  (2, 2)))
model.add(tf.keras.layers.Convolution2D(filters=50, kernel_size = (5, 5), padding = "same"))
model.add(tf.keras.layers.Activation(activation= "relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides = (2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500))
model.add(tf.keras.layers.Activation(activation = "relu"))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation("softmax"))

model.summary()

model.compile(optimizer='adam', #tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
              #loss='categorical_crossentropy',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

# COMMAND ----------

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

%time history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, \
                          validation_data=(X_test, y_test), callbacks=[early_stopping])

model.evaluate(X_test, y_test, verbose=1)

# COMMAND ----------

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

fig = plt.figure(figsize=(14,6))
plt.style.use('bmh')
params_dict = dict(linestyle='solid', linewidth=0.25, marker='o', markersize=6)

plt.subplot(121)
plt.plot(hist_df.loss, label='Training loss', **params_dict)
plt.plot(hist_df.val_loss, label='Validation loss', **params_dict)
plt.title('Loss for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(hist_df.accuracy, label='Training accuracy', **params_dict)
plt.plot(hist_df.val_accuracy, label='Validation accuracy', **params_dict)
plt.title('Accuracy for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ### d) Testing a more complex model
# MAGIC 
# MAGIC ![](https://sihamtabik.github.io/LeNet-like-CNN.jpg)

# COMMAND ----------

nb_filters_1 = 32 # 64
nb_filters_2 = 64 # 128
nb_filters_3 = 128 # 256
nb_conv = 3

model = tf.keras.Sequential()
model.add(tf.keras.layers.ZeroPadding2D((1,1), input_shape=(28, 28, 1),))
model.add(tf.keras.layers.Conv2D(nb_filters_1, (nb_conv, nb_conv),  activation="relu"))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters_1, (nb_conv, nb_conv), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(strides=(2,2)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters_2, (nb_conv, nb_conv), activation="relu"))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters_2, (nb_conv, nb_conv), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(strides=(2,2)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters_3, (nb_conv, nb_conv), activation="relu"))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters_3, (nb_conv, nb_conv), activation="relu"))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters_3, (nb_conv, nb_conv), activation="relu"))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters_3, (nb_conv, nb_conv), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer='adam', #tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
              #loss='categorical_crossentropy',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

# COMMAND ----------

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])
model.evaluate(X_test, y_test, verbose=1)

# COMMAND ----------

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

fig = plt.figure(figsize=(14,6))
plt.style.use('bmh')
params_dict = dict(linestyle='solid', linewidth=0.25, marker='o', markersize=6)

plt.subplot(121)
plt.plot(hist_df.loss, label='Training loss', **params_dict)
plt.plot(hist_df.val_loss, label='Validation loss', **params_dict)
plt.title('Loss for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(hist_df.accuracy, label='Training accuracy', **params_dict)
plt.plot(hist_df.val_accuracy, label='Validation accuracy', **params_dict)
plt.title('Accuracy for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pre-trained models  
# MAGIC 
# MAGIC How can I use pre-trained models in Keras?  
# MAGIC Code and pre-trained weights are available for the following image classification models:  
# MAGIC + Xception  
# MAGIC + VGG16  
# MAGIC + VGG19  
# MAGIC + ResNet50  
# MAGIC + Inception v3  
# MAGIC 
# MAGIC They can be imported from the module [keras.applications](https://keras.io/applications/):

# COMMAND ----------

#from tf.keras.applications.xception import Xception
#from tf.keras.applications.vgg16 import VGG16
#from tf.keras.applications.vgg19 import VGG19
#from tf.keras.applications.resnet50 import ResNet50
#from tf.keras.applications.inception_v3 import InceptionV3

#model = VGG16(weights='imagenet', include_top=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Where can i go further?
# MAGIC [Transfer Learning and Fine-Tuning](https://keras.io/guides/transfer_learning/)  
# MAGIC [Visualizing the classification task](http://scs.ryerson.ca/~aharley/vis/fc/)
