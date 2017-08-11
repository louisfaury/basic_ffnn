# -*- coding: utf-8 -*-

''' Header '''
import numpy as np
import matplotlib.pyplot as plt

''' Consts '''
data_file = "../src/dataset/sin_dataset.txt"

''' Script '''
" reading and plotting data"
iter = -1
with open(data_file) as f:
    for line in f:
        if (iter<0):
            L = np.asarray([int(s) for s in line.split(',')])
            inSize = L[0]
            outSize = L[1]
            numSamples = L[2]
            inputs = np.zeros(shape = (int(numSamples),int(inSize)))
            outputs = np.zeros(shape = (int(numSamples),int(outSize)))
        else:
             L = [float(s) for s in line.split(',')]
             inputs[iter,:] = L[0:inSize]
             outputs[iter,:] = L[inSize:]
        iter = iter+1
plt.scatter(inputs,outputs)
" ploting function " "TODO switch case between functions"
x = np.linspace(-5,5,1000)
y = np.sin(x)
plt.plot(x,y)
