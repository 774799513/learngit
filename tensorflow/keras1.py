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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.utils.vis_utils import plot_model
x = np.linspace(-1,1,200)
y = 0.5*x+2+np.random.normal(0,0.05,200)

x_train,y_train =x[:160],y[:160]
x_test,y_test = x[160:],y[160:]


model = Sequential()
model.add(Dense(1,input_dim=1))

model.compile(loss='mse',optimizer='sgd')
print('Training -----------')
for step in range(5001):
    cost = model.train_on_batch(x_train, y_train) # Keras有很多开始训练的函数，这里用train_on_batch（）
    if step % 100 == 0:
        print('train cost: ', cost)
# 测试训练好的模型
print('\nTesting ------------')
cost = model.evaluate(x_test, y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()    # 查看训练出的网络参数
                                        # 由于我们网络只有一层，且每次训练的输入只有一个，输出只有一个
                                        # 因此第一层训练出Y=WX+B这个模型，其中W,b为训练出的参数
print('Weights=', W, '\nbiases=', b)
y_pred = model.predict(x_test)
print(x_test)
plt.axis([0.6,1,1.5,3])
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred)
plt.show()
