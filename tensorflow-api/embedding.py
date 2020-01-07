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
from tensorflow.keras.layers import Embedding
model = tf.keras.Sequential()
model.add(Embedding(2, 2, input_length=7))#输入维，输出维
model.compile('rmsprop', 'mse')
a = model.predict(np.array([[0,1,0,1,1,0,0]]))
print(a)
w = model.layers[0].W.get_value()
print(w)
