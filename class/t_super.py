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

class Base:
    def __init__(self,a,b):
        self.a = a
        self.b = b
        print(self.a,self.b)
class A(Base):
    def __init__(self,a,b,c):
        super().__init__(a,b)
        self.c=c
        print(self.a,self.b,self.c)


a = Base(1,2)
b = A(3,4,5)


class A:
    def m(self):
        print('A')

class B:
    def m(self):
        print('B')
class C(A):
    def m(self):
        super().m()
        print('C')
C().m()

