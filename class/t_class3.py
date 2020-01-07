#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from pathlib import Path
#import random,math,sympy
#import re,request
#from turtle import *
#import time,datetime
#import argparse
#F2:tree F3:tagbar  F4:添加注释 F5:run F8:Autopep8 F10:save&&exit

class Dog:
    def __init__(self,name):
        self.name =name
        print('this is a dog named',self.name)
    def __str__(self):
        return 'dogs'
dang = Dog('dang')
print(dang)
