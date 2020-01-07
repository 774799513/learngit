import tensorflow as tf
#import matplotlib.pyplot as plt
#import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy,random
#import time,os,sys,datetime 
#import requests,re
#with open    as file:
#from turtle import *
#from bs4 import BeautifulSoup
#with tf.Session() as sess:
#from keras.models import Sequential 
#from keras.layers import Dense,Activation,Dropout
#from keras.optimizer import *
# import IPhthon.display as display

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
# Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    # display.display(display.Image(data=image_raw))

