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

filename = '/home/syl//.keras/datasets/flower_photos/sunflowers/1008566138_6927679c8a.jpg'
image = tf.read_file(filename)
print(repr(image))
#变为tensor
image_jpeg = tf.image.decode_png(image,channels=3,name="decode_image")
print(image_jpeg.shape)#显示不了
print(image_jpeg.dtype)
image_jpeg = tf.reshape(image_jpeg, shape=(192,192,3))
img_shape = image_jpeg.shape
print(img_shape)
# 获取图片shape数据
width = img_shape[0]
height = img_shape[1]
# 将原图片tensor生成bytes对象， image将保存到tfrecord
sess = tf.Session()
image = sess.run(image)
sess.close()
# 定义TFRecords文件的保存路径及其文件名
path_none = "/home/syl/img_compress_none.tfrecord"
path_zlib = "/home/syl/img_compress_zlib.tfrecord"
path_gzip = "/home/syl/img_compress_gzip.tfrecord"

# TFRecords文件的压缩选项
options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

# 定义不同压缩选项的TFRecordWriter
writer_none = tf.python_io.TFRecordWriter(path_none, options=None)
writer_zlib = tf.python_io.TFRecordWriter(path_zlib, options=options_zlib)
writer_gzip = tf.python_io.TFRecordWriter(path_gzip, options=options_gzip)

# 编码数据，将数据生成特定类型列表后再编码为feature格式，再生成字典形式
feature_internal_none = {
"float_val":tf.train.Feature(float_list=tf.train.FloatList(value=[9.99])),
"width":tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
"height":tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
"image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
}
feature_internal_zlib = {
"float_val":tf.train.Feature(float_list=tf.train.FloatList(value=[8.88])),
"width":tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
"height":tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
"image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
}
feature_internal_gzip = {
"float_val":tf.train.Feature(float_list=tf.train.FloatList(value=[6.66])),
"width":tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
"height":tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
"image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
}
# 编码内层字典形式数据
features_extern_none = tf.train.Features(feature=feature_internal_none)
features_extern_zlib = tf.train.Features(feature=feature_internal_zlib)
features_extern_gzip = tf.train.Features(feature=feature_internal_gzip)
# 将外层features生成特定格式的example
example_none = tf.train.Example(features=features_extern_none)
example_zlib = tf.train.Example(features=features_extern_zlib)
example_gzip = tf.train.Example(features=features_extern_gzip)
# example系列化字符串
example_str_none = example_none.SerializeToString()
example_str_zlib = example_zlib.SerializeToString()
example_str_gzip = example_gzip.SerializeToString()
# 将系列化字符串写入协议缓冲区
writer_none.write(example_str_none)
writer_zlib.write(example_str_zlib)
writer_gzip.write(example_str_gzip)
# 关闭TFRecords文件操作接口
writer_none.close()
writer_zlib.close()
writer_gzip.close()
print("finish to write data to tfrecord file!")
