import sys,os,io,re 
import networkx as nx
from pathlib import Path
import numpy as np 
import pandas as pd
import tarfile
import itertools as it
import glob
import tensorflow as tf
from multiprocessing import Pool
import argparse
from random import shuffle
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def ned2lists(fname):
    '''
    Args:
        fname a textFile path
    Returns:
        connections - list of lists neighbors for each node
        n - number of nodes
        edges - a list of pars (srd,dst)
        capacities - link capcities
    '''
    channels = []
    capacities = []

    p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel(\d+)kbps\s+<-->\s*node(\d+).port\[(\d+)\]')
    with open(fname) as fobj:
        for line in fobj:
            m=p.match(line)
            if m:
                matches = list(map(int,m.groups()))
                capacities.append(matches[2])
                del matches[2]
                channels.append(matches)
    n=max(map(max, channels))+1
    
    connections = [{} for i in range(n)]
    for c in channels:
        connections[c[0]][c[1]]=c[2]
        connections[c[2]][c[3]]=c[0]
    connections = [[v for k,v in sorted(con.items())] 
                   for con in connections ]
    # edges = [(c[0],c[2]) for c in channels] + [(c[2],c[0]) for c in channels]
    A = np.zeros((n,n))
    for a,c in zip(A,connections):
        a[c]=1
    G = nx.from_numpy_array(A,create_using=nx.DiGraph)
    # nx.draw(G,with_labels=True,font_weight='bold')
    # plt.show()
    edges = list(G.edges)
    capacities = capacities + capacities
    return connections,n,edges,capacities


def load_routing(routing_file):
    '''
    Loads routing descriptions
    Args:
        routing_file: a file handle
    '''
    R = pd.read_csv(routing_file, header=None, index_col=False)
    R=R.dropna(axis=1)
    return R.values


def genPath(R,s,d,connections):
    while s != d:
        yield s
        s = connections[s][R[s,d]]
    yield s

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def make_indices(paths):
    link_indices=[]
    path_indices=[]
    sequ_indices=[]
    segment=0
    for p in paths:
        link_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment +=1
    return link_indices, path_indices, sequ_indices


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class UPCDataset:
    '''
    Helper converter of UPC datasets format to tfrecords
    Supports UPC dataset v2 (zip of tar.gz files)
    '''
        
    def __init__(self, ned_file, output_dir):
        (self.connections,self.n,self.edges,self.capacities) = ned2lists(ned_file)
        self.output_dir = output_dir
        
        
    def __call__(self, tarpath):
        '''
        Reads selected dataset from a zipped tar file and save it as tfrecord file
        '''
        logging.info('Started %s...',tarpath)
        # tarname = os.path.split(tarpath)[1]
        # tarpath = Path(tarpath)
        tarname = tarpath.stem
        # routing_name = tarname.replace('.tar.gz','/Routing.txt')
        routing_name = tarname.replace('.tar','/Routing.txt')

        # 有改动
        # data_name = tarname.replace('.tar.gz','/simulationResults.txt')
        data_name = tarname.replace('.tar','/simulationResults.txt')
        # tfrecords_name = os.path.join(self.output_dir,tarname.replace('tar.gz','tfrecords'))
        # output_dir = Path(output_dir)
        tfrecords_name = self.output_dir/(tarname.replace('tar','tfrecords'))
        tfrecords_name = str(tfrecords_name)
        with tarfile.open(tarpath) as tar:
            with tar.extractfile(routing_name) as fobj:
                R = load_routing(fobj)
            with tar.extractfile(data_name) as fobj:
                names, drop_names = self._make_names()
                data = pd.read_csv(fobj,header=None, names=names,index_col=False)
                data = data.drop(drop_names, axis=1)
        
        paths = self._make_paths(R)
        link_indices, path_indices, sequ_indices = make_indices(paths)
        
        delay = data.filter(regex='average_delay').values
        logdelay = data.filter(regex='average_log').values
        tm = data.filter(regex='traffic').values
        jitter = data.filter(regex='variance').values
        drops =  data.filter(regex='drops').values
        packets =  data.filter(regex='packets').values

        n_paths = delay.shape[1]
        n_links = max(max(paths))+1
        n_total = len(path_indices)

        writer = tf.io.TFRecordWriter(tfrecords_name)
        
        for item in zip(tm,delay,jitter,drops,packets,logdelay):
            example = tf.train.Example(features=tf.train.Features(feature={
            'traffic':_float_features(item[0]),
            'delay':_float_features(item[1]),
            'jitter':_float_features(item[2]),
            'drops':_float_features(item[3]),
            'packets':_float_features(item[4]),
            'logdelay':_float_features(item[5]),
            'links':_int64_features(link_indices),
            'paths':_int64_features(path_indices),
            'sequences':_int64_features(sequ_indices),
            'n_links':_int64_feature(n_links), 
            'n_paths':_int64_feature(n_paths),
            'n_total':_int64_feature(n_total),
            'capacities':_float_features(self.capacities)

            }))
            
            writer.write(example.SerializeToString())
        writer.close()
        logging.info('Finished %s...',tarpath)

        
    def _make_paths(self, R):
        '''
        Construct path description for a given omnet++ routing
        Args:
            R - Routing matrix in omnet++ format
        Returns:
            A list of link indices for each path
        '''
        paths=[]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    paths.append([self.edges.index(tup) for tup in pairwise(genPath(R,i,j,self.connections))])
        return paths

    def _make_names(self):
        '''
        Generate names for dataset columns
        '''
        n=self.n
        names=[]

        counter=0
        drop_names=[]

        for i in range(n):
            for j in range(n):
                names.append('traffic_{}_{}'.format(i,j))
                if i == j: drop_names.append(names[-1])
                counter += 1
                names.append('packets_{}_{}'.format(i,j))
                if i == j: drop_names.append(names[-1])
                counter += 1
                names.append('drops_{}_{}'.format(i,j))
                if i == j: drop_names.append(names[-1])
                counter += 1
        for i in range(n):
            for j in range(n):
                for k in ['average','average_log' ,'q10','q20','q50','q80','q90','variance']:
                    names.append('{}_delay_{}_{}'.format(k,i,j))
                    if i == j: drop_names.append(names[-1])
                    counter += 1
        names.append('empty')
        return names, drop_names
                        
                    
DESCRIPTION='''
UPC dataset processor
Helper command to unzip files:
 find . -name '*.zip' | xargs -P 4 -L 1 unzip -u
'''
def main():
    p = Path('.')
    with Pool(processes=2) as pool:
        ned_file = list(p.glob('*.ned'))[0] 
        tars = list(p.glob('*.tar.gz'))
        shuffle(tars)
        first_n = int(0.7*len(tars))
        # print(len(tars))
        # print(first_n)
        processor = UPCDataset(ned_file,p/'train')
        # pool.map(processor,tars[0:first_n])
        pool.map(processor,tars[:first_n])
        processor.output_dir = p/'evaluate'
        pool.map(processor,tars[first_n:])
if __name__ == '__main__':
    main()
