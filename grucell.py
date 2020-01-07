import tensorflow as tf
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

 
batch_size=10
 
depth=128
 
output_dim=100
 
inputs=tf.Variable(tf.random_normal([batch_size,depth]))
 
previous_state=tf.Variable(tf.random_normal([batch_size,output_dim])) #前一个状态的输出
 
gruCell=tf.nn.rnn_cell.GRUCell(output_dim)
 
output,state=gruCell(inputs,previous_state)
 
print(output)
 
print(state)

