#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt
#from pathlib import Path
#import random,math
#import re

import argparse
parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('integers',metavar='N',type=int,nargs='+',help='an integer for the accumulator')
parser.add_argument('--sum',dest='accumulate',action='store_const',const=sum,default=max,help='sum the integers')
args = parser.parse_args()
print(args.accumulate(args.integers))

