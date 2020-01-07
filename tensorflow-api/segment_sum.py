import tensorflow as tf
import numpy as np
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
#from tensorflow import keras
#from tensorflow.keras import layers

ids = np.array([0,0,0,1,1])
data = np.random.randint(1,100,5)
print(data)
c = tf.math.segment_sum(data,ids)
print(c)

ids2 = np.array([0,1,0,1,0])
data2 = np.random.randint(1,90,5)
print(data2)
d = tf.math.unsorted_segment_sum(data2,ids2,num_segments=2)
print(d)
data3 = np.random.randint(1,20,20).reshape(5,4)
print(data3)
id3 = np.array([0,1,2,1,1])
f = tf.math.unsorted_segment_sum(data3,id3,num_segments=5)
print(f)
