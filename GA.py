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
    

 #*10是為了方便做二進位轉換，最後再除回來 (好像不需要了)
a = bin(50)
int(a, 2)


#找到最大、最小值了接資料的區間

maxIris = np.max(X)
minIris = np.min(X)

# 進位轉換
def ToBinary(numDecimal): # 將 iris 資料轉二進位，再進入 GA
    toBin = round(((numDecimal + 1) * (2 ** 10 - 1)) / 8)
    toBin = bin(toBin)
    return toBin

def ToDecimal(binNum): # 從 GA 進化出來的二進位資料，轉換成根據範圍限制的實際的數字
    toDec = int(binNum, 2)
    toDec = -1 + toDec * (8 / (2 ** 10 - 1))
    return toDec

# Parent 從 iris 中隨機選擇
def Parent():
    
    return

# Fitness 計算，依照不同需求會有不同的 fitness 計算，這裡求距離最小（詳見手稿）。 Fitness 會影響配對選擇之機率。
def Fitness():
    
    return

# ParentPair 湊成對 → Parents
def ParentPair():
    return

# Crossover 交配，進入交配前先將資料轉二進位
def Crossover():
    return

# Mutation 突變，突變後資料要轉回十進位
def Mutation():
    return

# KMeans 根據突變結果計算分類正確率，若未達標從 Fitness 繼續下一輪
def KMeans():
    return