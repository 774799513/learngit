import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy,random
#import requests,re
#with open() as file:
#from turtle import *
#from bs4 import BeautifulSoup
#with tf.Session() as sess:
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,Activation,Dropout
#from tensorflow.optimizer import *
#import time,os,sys,datetime  cwd = os.getcwd()
#from pathlib import Path p=Path('.')
#F2:tree F4:notes  F5:run F8:pep8 F3:tagbar,F10:save
from tensorflow.keras import layers
# build model
model = tf.keras.Sequential()
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation="softmax"))
# compile
model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
            loss = tf.keras.losses.categorical_crossentropy,
            metrics = [tf.keras.metrics.categorical_accuracy]
        )

# import numpy data
# train data
train_x = np.random.random((1000,72))
train_y = np.random.random((1000,10))
# valid data
val_x = np.random.random((200,72))
val_y =np.random.random((200,10))

model.fit(train_x,train_y,epochs=10,batch_size=100,
        validation_data=(val_x,val_y))
# use tf.data data
dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
dataset =dataset.batch(32)
dataset =dataset.repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x,val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()

model.fit(dataset,epochs =10,steps_per_epoch=30,
        validation_data=val_dataset,validation_steps=3)
# evaluate and predict
text_x = np.random.random((1000,72))
test_y = np.random.random((1000,10))
model.evaluate(test_x,test_y,batch_size=32)
test_data = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_data = test_data.batch(32).repeat()

model.evaluate(test_data,steps=30)
#predict
result = model.predict(test_x,batch_size=32)
print(result)


