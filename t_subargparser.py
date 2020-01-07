#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt
#from pathlib import Path
#import random,math,sympy
#import re,request
#from turtle import *
#import time,datetime
import argparse
#F2:tree F3:tagbar  F4:添加注释 F5:run F8:Autopep8 F10:save&&exit
def foo(args):
    print('song yuan long is genius:',args.g,args.d)
parser = argparse.ArgumentParser(description='this is a description')
subparsers = parser.add_subparsers(help='sub-command help')
create_parser = subparsers.add_parser('create',help='songyuanlong create')
create_parser.add_argument('-g',help='create',type=str)
create_parser.add_argument('-d',help='dir',type=str)
create_parser.set_defaults(func=foo)

args = parser.parse_args()
args.func(args)
# print(args)
