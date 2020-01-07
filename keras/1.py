import tensorflow as tf
import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from pathlib import Path
#import random,math,sympy
#import re,request
#from turtle import *
#import time,datetime
#import argparse
#matplotlib.use('TkAgg')

from tensorflow.keras import layers
# print(tf.__version__)
# print(tf.keras.__version__)


train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))


model = tf.keras.Sequential()
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy])
model.fit(train_x, train_y, epochs=10, batch_size=100,
          validation_data=(val_x, val_y))
