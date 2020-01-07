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

ids = np.array([[1],[3],[5],[7]])
updates = np.random.randint(1,10,4)
shape = tf.constant([10])
s = tf.scatter_nd(ids,updates,shape)
print(s)


id2 = np.array([[0],[2]])
updates2 = np.array([
                     [[5,5,5,5],
                      [6,6,6,6],
                      [7,7,7,7],
                      [8,8,8,8]],
                     [[0,0,0,0],
                      [1,1,1,1],
                      [2,2,2,2],
                      [3,3,3,3]]
                      ])
shape2 = tf.constant([4,4,4])
s2 = tf.scatter_nd(id2,updates2,shape2)
print(s2)

id3 = np.array([[0,0],[1,2]])
updates3 = np.array([
                    [3,3,3,3],
                    [4,4,4,4]
                    ])

shape3 = tf.constant([5,3,4])
s3 = tf.scatter_nd(id3,updates3,shape3)
print(s3)
