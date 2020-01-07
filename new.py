import tarfile
import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool
# from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
# import random,math,sympy
import re
# from turtle import *
# import time,datetime
# import argparse
# F2:tree F3:tagbar  F4:添加注释 F5:run F8:Autopep8 F10:save&&exit
import itertools as it
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def ned2list2(fname):
    channels = []
    capacities = []
    p = re.compile(
        r'\s+node(\d+).port\[(\d)\]\s+<-->\s+Channel(\d+)kbps\s+<-->\s*node(\d+).port\[(\d)\]')
    with open(fname) as fobj:
        for line in fobj:
            m = p.match(line)
            if m:
                matches = list(map(int, m.groups()))
                capacities.append(matches[2])
                del matches[2]
                channels.append(matches)
    print(channels)
    print('channelslen', len(channels))
    n = max(map(max, channels))+1
    connections = [{} for i in range(n)]
    for c in channels:
        connections[c[0]][c[1]] = c[2]
        connections[c[2]][c[3]] = c[0]
    print('connections1', connections)
    connections = [[v for k, v in sorted(con.items())]
                   for con in connections]
    A = np.zeros((n, n))
    for a, c in zip(A, connections):
        a[c] = 1
    # print('A')
    # print(A)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    # nx.draw(G,with_labels=True,font_weight='bold')
    # plt.show()
    edges = list(G.edges)
    capacities = capacities + capacities
    return connections, n, edges, capacities


p = Path('.')
tars = list(p.glob('*.tar.gz'))
ned_file = list(p.glob('*.ned'))[0]
connections, n, edges, capacities = ned2list2(ned_file)
print('connections')
print(connections)
print('n')
print(n)
print('edges')
print(edges)
# for k,v in enumerate(edges):
# print(k,v)
# print(len(edges))

print('capacities')
print(capacities)


def load_routing(routing_file):
    R = pd.read_csv(routing_file, header=None, index_col=False)
    R = R.dropna(axis=1)
    return R.values


routing_file = p/'results_nsfnetbw_9_Routing_SP_k_8'/'Routing.txt'
R = load_routing(routing_file)
# print(R)


def genPath(R, s, d, connections):
    while s != d:
        yield s
        s = connections[s][R[s, d]]
    yield s


def pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def make_indices(paths):
    link_indices = []
    path_indices = []
    sequ_indices = []
    segment = 0
    for p in paths:
        link_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment += 1
    return link_indices, path_indices, sequ_indices
# q = p/'results_nsfnetbw_9_Routing_SP_k_8'/'Routing.txt'
# print(q)
# temp = load_routing(q)
# print(temp)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def make_paths(R, connections, edges, n):
    paths = []
    for i in range(n):
        for j in range(n):
            if i != j:
                paths.append([edges.index(tup)
                              for tup in pairwise(genPath(R, i, j, connections))])
                # print(i,j,'paths',paths)
        # print('paths')
        # print(paths)
    return paths


paths = make_paths(R, connections, edges, n)
print('path', paths)
link_indices, path_indices, sequ_indices = make_indices(paths)
print('links', link_indices, len(link_indices))
print('paths', path_indices, len(path_indices))
print('seqs', sequ_indices, len(sequ_indices))
print('n_path', len(paths))

shape = tf.stack([42,3],axis=0)
capacities = np.array(capacities)
# capacities = capacities+0.0
capacities = tf.cast(capacities,tf.float32)
# print(capacities)
link_state = tf.concat([tf.expand_dims(capacities,axis=1),tf.zeros(shape)],axis=1)
# print(link_state)
h_ = tf.gather(link_state,link_indices)
print('h_',h_)
ids = tf.stack([path_indices,sequ_indices],axis=1)
print('ids',ids)

max_len = tf.reduce_max(sequ_indices)+1
# print('max_len',max_len)
shape = tf.stack([len(paths),max_len,4],axis=0)
# print(shape)
paths = np.array(path_indices)
print('paths')
print(paths)
lens = tf.math.segment_sum(data=tf.ones_like(paths),
                            segment_ids=paths)
print('lens')
print(lens)
link_inputs = tf.scatter_nd(ids,h_,shape)
# print(link_inputs)

# outputs,path_state = tf.






def make_names(n):
    '''
    Generate names for dataset columns
    '''
    names = []

    counter = 0
    drop_names = []

    for i in range(n):
        for j in range(n):
            names.append('traffic_{}_{}'.format(i, j))
            if i == j:
                drop_names.append(names[-1])
            counter += 1
            names.append('packets_{}_{}'.format(i, j))
            if i == j:
                drop_names.append(names[-1])
            counter += 1
            names.append('drops_{}_{}'.format(i, j))
            if i == j:
                drop_names.append(names[-1])
            counter += 1
    for i in range(n):
        for j in range(n):
            for k in ['average', 'average_log', 'q10', 'q20', 'q50', 'q80', 'q90', 'variance']:
                names.append('{}_delay_{}_{}'.format(k, i, j))
                if i == j:
                    drop_names.append(names[-1])
                counter += 1
    names.append('empty')
    return names, drop_names


tarpath = tars[0]
print('tarpath', tarpath)
tarname = tarpath.stem
routing_name = tarname.replace('.tar', '/Routing.txt')
data_name = tarname.replace('.tar', '/simulationResults.txt')
# tfrecords_name = self.output_dir/(tarname.replace('.tar','tfrecords'))

with tarfile.open(tarpath) as tar:
    with tar.extractfile(routing_name) as fobj:
        R = load_routing(fobj)
    with tar.extractfile(data_name) as fobj:
        names, drop_names = make_names(n)
        data = pd.read_csv(fobj, header=None, names=names, index_col=False)
        data = data.drop(drop_names, axis=1)
    # paths = make_paths(R, connections)
    # link_indices, path_indices, sequ_indices = make_indices(paths)


    delay = data.filter(regex='average_delay').values
    logdelay = data.filter(regex='average_log').values
    tm = data.filter(regex='traffic').values
    jitter = data.filter(regex='variance').values
    drops = data.filter(regex='drops').values
    packets = data.filter(regex='packets').values
    # n_paths = delay.shape[1]
    # n_links = max(max(paths))+1
    # n_total = len(path_indices)
# print(n_links)
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
capacities = _float_features(capacities)

shape = tf.stack([n_links,3],axis=0)
paths = _int64_features(path_indices)
# lens = tf.math.segment_sum(data=tf.ones_like(paths),segment_ids=paths)
# print(lens)
# link_state=tf.concat([tf.expand_dims(capacities,axis=1),tf.zeros(shape,dtype=float)],axis=1)
# print(link_state)
# print(n_paths)<F4>
        # writer = tf.io.TFRecordWriter(tfrecords_name)

        # for item in zip(tm,delay,jitter,drops,packets,logdelay):
        # example = tf.train.Example(features=tf.train.Features(feature={
        # 'traffic':_float_features(item[0]),
        # 'delay':_float_features(item[1]),
        # 'jitter':_float_features(item[2]),
        # 'drops':_float_features(item[3]),
        # 'packets':_float_features(item[4]),
        # 'logdelay':_float_features(item[5]),
        # 'links':_int64_features(link_indices),
        # 'paths':_int64_features(path_indices),
        # 'sequences':_int64_features(sequ_indices),
        # 'n_links':_int64_feature(n_links),
        # 'n_paths':_int64_feature(n_paths),
        # 'n_total':_int64_feature(n_total),
        # 'capacities':_float_features(self.capacities)

        # }))

        # writer.write(example.SerializeToString())
        # writer.close()
        # logging.info('Finished %s...',tarpath)
# with Pool(processes=2) as pool:
