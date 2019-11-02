#change
#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy,random
#import requests,re
#with open() as file:
#from turtle import *
#from bs4 import BeautifulSoup
#with tf.Session() as sess:
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,Activation,Dropout
#from tensorflow.optimizer import *
#import time,os,sys,datetime  cwd = os.getcwd()
#from pathlib import Path p=Path('.')
#F2:tree F4:notes  F5:run F8:pep8 F3:tagbar,F10:save
import argparse
def main():
    parser = argparse.ArgumentParser(description="Demo of argparse") 
    parser.add_argument('-d',help='Data directiory', type=str,required=True,nargs='+') 
    parser.add_argument('--processes',help='number of concurrent jobs',default=2,type=int) 
    args = parser.parse_args() 
    for d in args.d:
        print(d)
    print(args) 
if __name__ == '__main__': main()
