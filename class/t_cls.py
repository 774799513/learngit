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
# classmethod 不用实例化类
class Circle():
    _pi = 3.14

    def __init__(self,r):
        self.r = r
    @classmethod
    def pi(cls):
        return cls._pi
    def area(self):
        return self.r**2*self._pi

print(Circle.pi())
circle1 = Circle(2)
print(circle1.pi())


class Date():
    day = 0
    month = 0
    year = 0

    def __init__(self,year=0,month=0,day=0):
        self.day =day
        self.month = month
        self.year = year
    @classmethod
    def from_string(cls,date_as_string):
        year,month,day = date_as_string.split('-')
        date = cls(year,month,day)
        return date
    # staticmethod接受外部参数
    @staticmethod
    def is_date_valid(date_as_string):
        year,month,day = date_as_string.split('-')
        return int(year)<2100 and int(month)<=12 and int(day)<32


date1 = Date.from_string('2019-10-10')
print(date1.year,date1.month,date1.day)
is_date = Date.is_date_valid('2012-02-03')
print(is_date)
