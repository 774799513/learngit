import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import math,sympy,random
# import time,os,sys,datetime
# import requests,re
# with open    as file:
# from turtle import *
# from bs4 import BeautifulSoup
# with tf.Session() as sess:
# from keras.models import Sequential
# from keras.layers import Dense,Activation,Dropout
# from keras.optimizer import *


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# the number of observations in the dataset
n_observations = int(1e4)
# boolean feature, encoded as False or True
feature0 = np.random.choice([False, True], n_observations)
# integer feature, random from 0 .. 4
feature1 = np.random.randint(0, 5, n_observations)
# string feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
# float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)


def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.Example message ready to be written to a file.
    """
# Create a dictionary mapping the feature name to the tf.Example-compatible
# data type.
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# This is an example observation from the dataset.
example_observation = []
serialized_example = serialize_example(False, 4, b'goat', 0.9876)
example_proto = tf.train.Example.FromString(serialized_example)
print(serialized_example)
example_proto = tf.train.Example.FromString(serialized_example)
print(example_proto)
# print(tf.data.Dataset.from_tensor_slices(feature1))
features_dataset = tf.data.Dataset.from_tensor_slices(
    (feature0, feature1, feature2, feature3))
# print(features_dataset)
for f0, f1, f2, f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)
def serialize_example_pyfunction(feature0, feature1, feature2, feature3):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'feature0': _int64_feature(feature0.numpy()),
        'feature1': _int64_feature(feature1.numpy()),
        'feature2': _bytes_feature(feature2.numpy()),
        'feature3': _float_feature(feature3.numpy()),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
        serialize_example_pyfunction,
        (f0, f1, f2, f3),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar
serialized_features_dataset = features_dataset.map(tf_serialize_example)
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

