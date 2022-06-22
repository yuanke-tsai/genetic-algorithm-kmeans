#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:46:05 2022

@author: yuanke.tsai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

data = datasets.load_iris()

X = data.data
Y = data.target

index = np.array([])
for i in range(30):
    indx = np.random.randint(0, 150)
    index = np.append(index, indx, axis=None)
    
parent = np.array([])
for p in range(len(index)):
    x = X[int(index[p])]
    #這邊做二進制轉換
    parent = np.append(parent, x, axis=None)
    


