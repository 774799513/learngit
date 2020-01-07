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

import itertools as it

def pairwise(iterable):
    a,b = it.tee(iterable)
    next(b,None)
    return zip(a,b)

a = [1,2,3,4]
pairwise(a)
for i in pairwise(a) :
    print()
