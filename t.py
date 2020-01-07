import tensorflow as tf
import numpy as np
import pandas as pd
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
#import time,datetime
from pathlib import Path 
p = Path('.')
#import glob,os,sys,cwd = os.getcwd()
#F2:tree F4:notes  F5:run F8:pep8 F3:tagbar,F10:save
import tarfile
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_routing(routing_file):
    R = pd.read_csv(routing_file,header=None,index_col=False)
    R=R.dropna(axis=1)
    return R.values
def make_name(n):
    n = n
    names = []
    # counter = 0
    drop_names = []
    for i in range(n):
        for j in range(n):
            names.append('traffic_{}_{}'.format(i,j))
            if i==j:drop_names.append(names[-1])
            # counter = counter + 1
            names.append('packets_{}_{}'.format(i,j))
            if i==j:drop_names.append(names[-1])
            # counter = counter + 1
            names.append('drops_{}_{}'.format(i,j))
            if i==j:drop_names.append(names[-1])
            # counter = counter + 1
    for i in range(n):
        for j in range(n):
            for k in ['average','average_log','q10','q20','q50','q90','variance']:
                names.append('{}_delay_{}_{}'.format(k,i,j))
                if i==j:drop_names.append(names[-1])
                # counter = counter+1
    names.append('empty')
    return names,drop_names

tarpath = 'results_nsfnetbw_9_Routing_SP_k_0.tar.gz'
outdir = p/'mytensorflow'
# outdir.mkdir()
logging.info('start %s ...',tarpath)
tarname = tarpath 
print('tarname:',tarname)
routing_name = tarname.replace('.tar.gz','/Routing.txt')
print('routing_name:',routing_name)
print('tarname:',tarname)
data_name = tarname.replace('.tar.gz','/simulationResults.txt')
print('data_name:',data_name)
tfrecords_name = outdir/(tarname.replace('tar.gz','tfrecords'))
print(tfrecords_name)

with tarfile.open(tarpath) as tar:
    with tar.extractfile(routing_name) as fobj:
        R = load_routing(fobj)
        # print(R) 
        n = R.shape[0]
        # print(n)
    with tar.extractfile(data_name) as fobj:
        names,drop_names =make_name(n)
        # print('names:',names)
        # print('drop_names:',drop_names)
        # print()
        data = pd.read_csv(fobj,header=None,names=names,index_col=False)
        data = data.drop(drop_names,axis=1)

        # print(data)
        tm = data.filter(like='traffic',axis=1).values
        # print('tm')
        # print(tm)
        # print(tm.shape)
        delay = data.filter(like='average_delay',axis=1).values
        logdelay = data.filter(like='average_log',axis=1).values
        # print('delay')
        # print(delay)
        # print(delay.shape)
        # print('logdelay')
        # print(logdelay)
        jitter = data.filter(regex='variance').values
        # print('jitter')
        # print(jitter)
        # print(jitter.shape)
        drops = data.filter(regex='drops').values
        # print('drops')
        # print(drops)
        packets = data.filter(regex='packets').values
        # print('packets')
        # print(packets)
        n_paths=delay.shape[1]
        # print(n_paths)
        n_links = 15
        n_total = 10
        writer = tf.io.TFRecordWriter(str(tfrecords_name))
        # for item in zip(tm,delay,jitter,drops,packets,logdelay):
            # example = tf.train.Example(features=tf.train.Freatures(feature={
                # 'traffic':

                # })
        pass
        writer.write(example.SerializeToString())
