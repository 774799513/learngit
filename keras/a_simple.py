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
#logger.setLevel(logging.INFO)
#logging.info('')
#model.summary
#keras.utils.plot_model(model,'model.png')
#keras.utils.plot_model(model,'model_info.png',show_shapes=True)
from tensorflow import keras
from tensorflow.keras import layers

class MyLayer(layers.Layer):
    def __init__(self,input_dim=32,unit=32):
        super(MyLayer,self).__init__()
        
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(
            shape = (input_dim,unit),dtype=tf.float32
            ),trainable=True)
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(
            shape=(unit,),dtype=tf.float32
            ),trainable=True)
    def call(self,inputs):
        return tf.matmul(inputs,self.weight)+self.bias

x = tf.ones((3,5))
my_layer = MyLayer(5,4)
out = my_layer(x)

print(out)

