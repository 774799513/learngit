#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy,random
#import requests,re
#with open() as file:
#from turtle import *
#from bs4 import BeautifulSoup
#with tf.Session() as sess:
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,Activation,Dropout
#from tensorflow.optimizer import *
#import time,datetime
from pathlib import Path 
p=Path('.')
#import glob,os,sys,cwd = os.getcwd()
#F2:tree F4:notes  F5:run F8:pep8 F3:tagbar,F10:save
# a = p.cwd()
# print(a)
# print(p.group())
q=p/'t1.py'
# with q.open()as f:
    # print(f.readlines())
# for i in p.iterdir():
    # print(i)
a = q.resolve()
print(a.name)
# print(a.parent)
# for i in p.glob('*'):print(i)
with q.open()as f:
    for line in f:
        print(line)
