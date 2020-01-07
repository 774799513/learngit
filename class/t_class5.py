#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from pathlib import Path
#import random,math,sympy
#import re,request
#from turtle import *
#import time,datetime
#import argparse
#matplotlib.use('TkAgg')
import math
class Log:
    def __init__(self,base=2):
        self.base = base
    def __call__(self,n):
        return math.log(n,self.base)

log10 = Log(10)
print(log10(10))


