import tensorflow as tf
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
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import *
np.random.seed(1)
data = np.random.random((1000,100))
labels = np.random.randint(0,2,(1000,1))
data_test = np.random.random((200,100))
labels_test = np.random.randint(0,2,(200,1))

model = Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])


model.fit(data,labels,epochs=500,batch_size=60)
loss, accuracy = model.evaluate(data_test, labels_test)
print('test loss: ',loss)
print('test accuracy: ',accuracy)
