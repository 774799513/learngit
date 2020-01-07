#import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy
#import time,os,sys, 
#import requests,re
#with open    as file:
#import turtle as tt
#from bs4 import BeautifulSoup
#with tf.Session() as sess:
from keras.models import Sequential 
from keras.layers import Dense,Activation
import keras
np.random.seed(1)
data = np.random.random((1000,100))
labels = np.random.randint(10,size =(1000,1))
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
data_test= np.random.random((200,100))
labels_test=np.random.randint(10,size=(200,1))
one_hot_labels_test = keras.utils.to_categorical(labels_test,num_classes=10)



model = Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(10,activation='softmax'))

model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

model.fit(data, one_hot_labels, epochs=200, batch_size=32)

loss, accuracy = model.evaluate(data_test,one_hot_labels_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
