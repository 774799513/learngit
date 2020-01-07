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

class Test:
    def __init__(self):
        pass
    def _one_underling(self):
        print('_one_underling')
    def __two_underling(self):
        print('two')

    def output(self):
        self._one_underling()
        self.__two_underling()
test = Test()
test.output()

