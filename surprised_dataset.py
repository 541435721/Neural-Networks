# coding:utf-8
# @Author:bianxuesheng

from numpy import *
from pybrain.datasets import SupervisedDataSet

DS = SupervisedDataSet(3, 1)
samples = array([[[1, 2, 3], [1]],
                 [[2, 3, 4], [2]],
                 [[3, 4, 5], [3]],
                 [[4, 5, 6], [4]]])
for inp, targ in samples:
    DS.appendLinked(inp, targ)

DS.addSample([5, 6, 7], [5])

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

DS.addField('myfield', 1)
myarray = array([[0.],
                 [0.],
                 [1.],
                 [1.],
                 [1.]])
DS.setField('myfield', myarray)
DS.linkFields(('input', 'target', 'myfield'))

