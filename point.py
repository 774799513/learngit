#import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy,random
#import requests,re
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
from shapely.geometry import Polygon
from shapely.geometry import Point
# poly = Polygon([(0,0,1),(0,0,-1),(1,0,-1),(1,0,1)])
point1 = Point(0.5, 0.5, 0.5)
point2 = Point(0, 0, 0)
d = point1.distance(point2)
# print(poly.contains(point))
print(d)
