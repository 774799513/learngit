import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from pathlib import Path
#import random,math,sympy
#import re
#from turtle import *
#import time,datetime
#import argparse
#import logging
#logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logging.info('')
# model.summary
# keras.utils.plot_model(model,'model.png')
# keras.utils.plot_model(model,'model_info.png',show_shapes=True)
from tensorflow import keras
from tensorflow.keras import layers


class ThreeLayerMlp(keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_1 = layers.Dense(64, activation='relu')
        self.dense_2 = layers.Dense(64, activation='relu')
        self.pred_layer = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.pred_layer(x)


def get_model():
    return ThreeLayerMlp(name='2_layer_mlp')


model = get_model()
model.summary()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
model.compile(loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,batch_size=64,epochs=1)
