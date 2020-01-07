#import tensorflow as tf
#import matplotlib.pyplot as plt
#import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import math,sympy
#import time,os,sys, 
#import requests,re
#with open    as file:
#import turtle as tt
import re
import requests
from bs4 import BeautifulSoup
import wxpy as wx
class SearchWeather():
    def __init__(self,cityCode,cityname):
        global weather
        self.cityCode = cityCode
        self.cityname = cityname
        self.HEADERS ={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 ''(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        url = 'http://www.weather.com.cn/weather/%s.shtml' % cityCode
        html = requests.get(url,headers = self.HEADERS)
        html.encoding='utf-8'
        soup=BeautifulSoup(html.text,'lxml')
        weather = "日期      天气    【温度】    风向风力\n"
        for item in soup.find("div", {'id': '7d'}).find('ul').find_all('li'):
            date,detail = item.find('h1').string, item.find_all('p')
            title = detail[0].string
            templow = detail[1].find("i").string
            temphigh = detail[1].find('span').string if detail[1].find('span')  else ''
            wind,direction = detail[2].find('span')['title'], detail[2].find('i').string
            if temphigh=='':
                weather += '你好，【%s】今天白天：【%s】，温度：【%s】，%s：【%s】\n' % (cityname,title,templow,wind,direction)
            else:
                weather += (date + title + "【" + templow +  "~"+temphigh +'°C】' + wind + direction + "\n")
SearchWeather('101120301','淄博'

bot = wx.Bot(cache_path = True)
family = bot.groups()
print(family)
bot.file_helper.send(weather)

