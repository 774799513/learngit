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
import math 
class Log:
    def __init__(self,base=2):
        self.base=base
    def __call__(self,number):
        return math.log(number,self.base)
log2 = Log(2)
log4 = Log(4)
print(log2(2))
print(log4(16))
