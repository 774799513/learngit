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

ids = np.array([[1,1],[0,0]])
data = np.array(
        [[2,3],
         [4,5]]
        )
out = tf.gather_nd(data,ids)
print(out)
