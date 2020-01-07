#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy,random
import requests
import re
# with open() as file:
#from turtle import *
#from bs4 import BeautifulSoup
# with tf.Session() as sess:
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,Activation,Dropout
#from tensorflow.optimizer import *
# import time,os,sys,datetime  cwd = os.getcwd()
# from pathlib import Path p=Path('.')
# F2:tree F4:notes  F5:run F8:pep8 F3:tagbar,F10:save
line = 'node0:Server{'
node_id_regex = re.compile('node(\d+):Server{')
m1 = node_id_regex.match(line)
current_node_id = int(m1.groups()[0])
print(current_node_id)
()
