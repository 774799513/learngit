#import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy,random
#import time,os,sys,datetime
#import requests,re
#with open() as file:
#from turtle import *
#from bs4 import BeautifulSoup
#with tf.Session() as sess:
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,Activation,Dropout
#from tensorflow.optimizer import *
#F2:tree F4:notes  F5:run F8:pep8 F9:tagbar,F10:save
from pathlib import Path 
p=Path('.')
subpath = [x for x in p.iterdir() if x.is_dir()]
# print(subpath)
py_files = list(p.glob('**/*.py'))
for py_file in py_files:
    print(py_file)
