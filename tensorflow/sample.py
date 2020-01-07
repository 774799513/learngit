import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy
#import time,os,sys, 
#import requests,re
#with open    as file:
#import turtle as tt
#from bs4 import BeautifulSoup
#with tf.Session() as sess:

x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.1,0.2],x_data)+0.3


b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(W,x_data)+b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train =optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

steps = []
Weight = []
bias = []

for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(W),sess.run(b))
