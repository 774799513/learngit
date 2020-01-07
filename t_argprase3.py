#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt
#from pathlib import Path
#import random,math
#import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d',type=str, required=True,nargs='+')
parser.add_argument('--processes',help='numbers',default=2,type=int)
args = parser.parse_args()
print(args)
print(args.d)
