#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt
#from pathlib import Path
#import random,math
#import re
import logging
from pathlib import Path
p = Path('.')

class stu:
    def __init__(self,ned_file,output_dir):
        self.ned_file=ned_file
        self.output_dir=output_dir
        print('init start')
    def __call__(self,tarpath):
        logging.info('started %s,,,',tarpath)
        tarname = tarpath.stem 
        # routing_name = tarname.replace()
        print('hello world',tarname)
a = stu('.','./bookmarks.html')
a(p/'songyuanlong')
