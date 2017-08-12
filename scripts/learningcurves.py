# -*- coding: utf-8 -*-

""" Header """
import numpy as np 
import matplotlib.pyplot as plt

""" Displaying learning curves """
# Importing data 
lglogfile = "../log/learninglogs.txt"
lgdata = np.genfromtxt(lglogfile,delimiter=',')
# Display 
iters = np.arange(0,(np.shape(lgdata))[0])
trainLoss = lgdata[:,0]
testLoss = lgdata[:,1]

f,(ax1,ax2) = plt.subplots(2,sharex=True)
ax1.plot(iters,trainLoss)
ax1.set_title('Training data loss')
ax2.plot(iters,testLoss,'r')
ax2.set_title('Testing data loss')
plt.savefig("../img/classic_learningcurves_2")
#plt.show()