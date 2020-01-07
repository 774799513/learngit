#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
#import random,math,sympy
import re
#from turtle import *
#import time,datetime
#import argparse
#matplotlib.use('TkAgg')

p = Path('.')
ned_file = list(p.glob('*.ned'))[0]
print(ned_file)
# regex = re.compile(r'\s+node')
regex = re.compile(r'\s+node(\d+).port\[(\d)\]\s+<-->\s+Channel(\d+)kbps\s+<-->\s*node(\d+).port\[(\d)\]')
channels = []
capacities = []
with open(ned_file) as f:
    for line in f:
        m = regex.match(line)
        if m:
            print(line)
            matches = list(map(int,m.groups()))
            capacities.append(matches[2])
            del matches[2]
            channels.append(matches)
print(channels)

