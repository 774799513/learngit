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
# unknown net input dim
class MyLayer(layers.Layer):
    def __init__(self,unit=32):
        super().__init__()
        self.unit=unit
    def build(self,input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1],self.unit),
                                      initializer = keras.initializers.RandomNormal(),
                                      trainable=True
                )
        self.bias = self.add_weight(shape=(self.unit,),
                                    initializer = keras.initializers.Zeros(),
                                    trainable=True
                )
    def call(self,inputs):
        return tf.matmul(inputs,self.weight)+self.bias

my_layer = MyLayer(3)
x =tf.ones((3,5))
out = my_layer(x)
print(out)

