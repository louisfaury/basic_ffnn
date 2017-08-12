# -*- coding: utf-8 -*-

""" Header """
import numpy as np 
import matplotlib.pyplot as plt

""" Displaying learning curves """
# Importing data 
simlogfile = "../log/simlog.txt"
simdata = np.genfromtxt(simlogfile,delimiter=',')
# Display 
plt.plot(simdata[:,0],simdata[:,1])
plt.plot(simdata[:,0],np.sin(simdata[:,0]),'r')