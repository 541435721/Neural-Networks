# coding:utf-8
# @Author:bianxuesheng

from numpy import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
DS = SupervisedDataSet(3, 1)

ia = array([[1.,  2.,  3.],
            [1.,  3.,  4.],
            [1.,  4.,  5.],
            [1.,  5.,  6.],
            [1.,  6.,  7.]])

ta = array([[5.],
            [4.],
            [3.],
            [2.],
            [1.]])
DS.setField('input', ia)
DS.setField('target', ta)

net = buildNetwork(3, 3, 1, bias=True)
trainer = BackpropTrainer(net, DS)
trainer.train()
print net.activate([1, 2, 3])
