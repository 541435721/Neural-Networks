# coding:utf-8
# @Author:bianxuesheng

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from numpy import *

net = FeedForwardNetwork()

input_level = LinearLayer(5, name='in')
hide1 = SigmoidLayer(3, name='hide1')
hide2 = LinearLayer(3, name='hide2')
hide3 = SigmoidLayer(2, name='hide3')
output_level = LinearLayer(1, name='out')

c1 = FullConnection(input_level, hide1, name='c1')
c2 = FullConnection(hide1, hide2, name='c2')
c3 = FullConnection(hide2, hide3, name='c3')
c4 = FullConnection(hide3, output_level, name='c4')

net.addInputModule(input_level)
net.addModule(hide1)
net.addModule(hide2)
net.addModule(hide3)
net.addOutputModule(output_level)

net.addConnection(c1)
net.addConnection(c2)
net.addConnection(c3)
net.addConnection(c4)

net.sortModules()

net1 = FeedForwardNetwork()
net2 = FeedForwardNetwork()

net1.addInputModule(input_level)
net1.addModule(hide1)
net1.addOutputModule(hide2)

net1.addConnection(c1)
net1.addConnection(c2)

net1.sortModules()

net2.addInputModule(hide2)
net2.addModule(hide3)
net2.addOutputModule(output_level)

net2.addConnection(c3)
net2.addConnection(c4)
net2.sortModules()

print net.activate([1, 2, 3, 4, 5])

mid = net1.activate([1, 2, 3, 4, 5])
print mid
print net2.activate(mid)
