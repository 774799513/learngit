#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt
from pathlib import Path
#import random,math,sympy
#import re,request
#from turtle import *
#import time,datetime
#import argparse
#F2:tree F3:tagbar  F4:添加注释 F5:run F8:Autopep8 F10:save&&exit

p = Path('.')
p = p.resolve()
print('cwd:',p.cwd())
filenames = list(p.glob('*.py'))
filenames2 = list(p.glob('??.py'))
print(filenames2)
for filename in filenames:
    print('filename:',filename)
    print('parent:',filename.parent)
    print('stem:',filename.stem)
    print('suffix:',filename.suffix)
