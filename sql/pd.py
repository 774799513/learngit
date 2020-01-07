#import tensorflow as tf
import numpy as np
import pandas as pd
#import networkx as nx
#import matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from pathlib import Path
#import random,math,sympy
#import re
#from turtle import *
#import time,datetime
#import argparse
#import logging
#logger = logging.getLogger()
#logger.setLevel(logging.INFO)
#logging.info('')
#model.summary
#keras.utils.plot_model(model,'model.png')
#keras.utils.plot_model(model,'model_info.png',show_shapes=True)
#from tensorflow import keras
#from tensorflow.keras import layers
from sqlalchemy import create_engine
conn_eng = create_engine('mysql+pymysql://syldb:uiojkl@localhost:3306/test')
dates = pd.date_range('20190101',periods=30)
df = pd.DataFrame(np.random.randn(30,4),index=dates,columns=list('ABCD'))
pd.io.sql.to_sql(df,'test1',conn_eng,if_exists='replace',index=True)
