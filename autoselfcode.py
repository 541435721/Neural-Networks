# coding:utf-8
# @Author:bianxuesheng

from numpy import *
import cv2
import time
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer

img = cv2.imread('test.jpg', 0)
# cv2.imshow('source', img)
temp = zeros((100, 100), dtype=float16)  # 源文件备份
temp = img
temp = temp.reshape(1, -1)  # 图像向量化

DS = SupervisedDataSet(100 * 100, 100 * 100)  # 数据集对象
DS.setField('input', temp)  # 初始化数据集输入
DS.setField('target', temp)  # 初始化数据集输出

net = FeedForwardNetwork()  # 创建前反馈网络

input_level = LinearLayer(100 * 100, name='in')  # 创建输入层
hide = LinearLayer(50 * 50, name='hide')  # 创建隐层
output_level = LinearLayer(100 * 100, name='out')  # 创建输出层

c1 = FullConnection(input_level, hide, name='c1')  # 创建输入到隐层连接
c2 = FullConnection(hide, output_level, name='c2')  # 创建隐层到输出层连接

net.addInputModule(input_level)  # 添加输出层
net.addModule(hide)  # 添加隐藏层
net.addOutputModule(output_level)  # 添加输出层

net.addConnection(c1)  # 添加c1连接
net.addConnection(c2)  # 添加c2连接

net.sortModules()  # 排列模块

trainer = BackpropTrainer(net, DS)  # 创建训练器
trainer.train()  # 训练网络

net1 = FeedForwardNetwork()  # 创建新网络1
net2 = FeedForwardNetwork()  # 创建新网络2

net1.addInputModule(input_level)  # 添加原网络的第一层
net1.addOutputModule(hide)  # 添加原网络的隐藏层作为新网络的输出层

net1.addConnection(c1)  # 添加原网络的第一层连接

net1.sortModules()

net2.addInputModule(hide)  # 原网络的隐藏层作为新网络2的输入层
net2.addOutputModule(output_level)  # 原网络的输出层作为新网络2的输出层

net2.addConnection(c2)  # 添加原网络的第二层链接

net2.sortModules()

code = net1.activate(temp[0].tolist())
# code = code.reshape(50, 50)
result = net2.activate(list(code))
cv2.imshow()
