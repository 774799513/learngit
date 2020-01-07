#import tensorflow as tf
import numpy as np
import pandas as pd
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

s = pd.Series([1,3,5,np.nan,6,8])
# 默认float64
print(s)

dates = pd.date_range('20130101',periods=30)
# 默认freq=day
# print(dates)

df = pd.DataFrame(np.random.randn(30,4),index=dates,columns=list('ABCD'))
# 数据，行索引，列索引
print(df)
# 默认前5个
print('head')
print(df.head())
print()
print('tail')
print(df.tail())
print(df.index)
print(df.columns)
print('summary')
print(df.describe())
print('转置')
print(df.T)


# index:axis=0
# columns:axis=1
print(df.sort_index(axis=1,ascending=False))
# 按b列生序
print(df.sort_values(by='B'))


print(df['A'])

print(df[0:3])

print(df['20130101':'20130105'])


# 按标签选择一行
print(df.loc[dates[0]])

print(df.loc[:,['A','B']])

print(df.loc['20130102':'20130109',['A','B']])
print()
print(df.loc['20130102',['A','B']])
print()
print(df.loc['20130103','A'])
# 按位置选择
print(df.iloc[3:5,0:2])
print(df.iloc[3])
print(df.iloc[[1,10,20],[0,2]])
print(df.iloc[1:3,:])
print(df.iloc[:,1:3])
print(df.iloc[1,1])
