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
from scipy import stats
import random

data = datasets.load_iris()
X = data.data
Y = data.target
# np.random.seed(1)
k = 3
crossoverRate = 0.9
mutationRate = 0.3

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
def Parent(dataX):
    idSelect = np.array([])
    parent = np.array([])
    for i in range(30):
        idx = int(np.random.randint(0, 150))
        idSelect = np.append(idSelect, idx, axis=None)
    idSelect = idSelect.astype(int, copy=False)
    
    for s in range(len(idSelect)):
        dataX = X[idSelect[s]]
        parent = np.append(parent, dataX, axis=None)
        s+=1
    parent = np.reshape(parent, (10, 3, 4))
    return parent

# Fitness 計算，依照不同需求會有不同的 fitness 計算，這裡求距離最小（詳見手稿）。 Fitness 會影響配對選擇之機率。
def Fitness(parent, X):
    fitness = np.array([])
    for j in range(len(parent)):
        for i in range(len(X)):
            eachParent_dis = 1/sum(np.sqrt(np.sum(np.square(parent[j] - X[i]), axis=1)))
        fitness = np.append(fitness, eachParent_dis, axis=None)
    # print("fitness len ", len(fitness))
    return fitness

# ParentPair 湊成對 → Parents
def ParentPair(fitness, parent):
    pair = np.array([])
    # print(np.shape(pair))
    for i in range(len(fitness)):
        # random.seed(i)
        single = random.choices(fitness, weights=fitness, k=1) # use random rather than np: https://blog.csdn.net/haha0332/article/details/115518985
        # print("single", single)
        # print(single, np.where(fitness == single))
        singleIdx = np.where(fitness == single)
        # print("singleIdx ", singleIdx[0][0])
        pair = np.append(pair, parent[singleIdx[0][0]], axis=None)
        i+=1
        # weighted choose (also can use cumulation)
        # https://pynative.com/python-weighted-random-choices-with-probability/
    # print("pair len ", len(pair))
    pair = np.reshape(pair, (5, 2, 3, 4)) 
    return pair

# Crossover 交配，進入交配前先將資料轉二進位
def Crossover(pair, crossoverRate):
    pairBin = np.array([])
    for i in range(len(pair)):
        for j in range(len(pair[0])):
            for k in range(len(pair[0][0])):
                for l in range(len(pair[0][0][0])):
                    Bin = ToBinary(pair[i][j][k][l])
                    pairBin = np.append(pairBin, Bin, axis=None)
    pairBin = np.reshape(pairBin, (5, 2, 3, 4))
    pairBinCopy = pairBin.copy()
    for i in range(int(len(fitness)/2)):
        # np.random.seed(i+3)
        threshold = np.random.rand()
        # print(threshold)
        if (threshold > crossoverRate):
            # print("s", i)
            pos = 7 # np.random.randint(0, 12) # 12數字需要再思考怎麼用參數
            for j in range(len(pairBin[0][0])):
                for k in range(len(pairBin[0][0][0])):
                    swap = pairBin[i][1][j][k][0:pos] + pairBin[i][0][j][k][pos:12]
                    pairBinCopy[i][0][j][k] = swap
                    swap2 = pairBin[i][0][j][k][0:pos] + pairBin[i][1][j][k][pos:12]
                    pairBinCopy[i][1][j][k] = swap2           
        # else: 好像不用
        #     print('wont change')
        # return 把原先的bin拿掉
    return pairBinCopy

# Mutation 突變，突變後資料要轉回十進位
def Mutation(pairBin, mutationRate):
    # 突變位置僅會有一個，2到12其一，變
    for i in range(len(pairBin)):
        for j in range(len(pairBin[0])):
            # np.random.seed(i+2)
            threshold = np.random.rand()
            # print(threshold)
            if threshold < mutationRate:
                pos_3 = np.random.randint(0, 3)
                pos_4 = np.random.randint(0, 4)
                # print('{}, {}, {}, {}' .format(i, j, pos_3, pos_4))
                pos_12 = np.random.randint(3, len(pairBin[i][j][pos_3][pos_4]))
                # print('pos_12: ', pos_12)
                if pairBin[i][j][pos_3][pos_4][pos_12] == '0':
                    # 直接用 xxx+1+xxx 
                    pairBin[i][j][pos_3][pos_4] = pairBin[i][j][pos_3][pos_4][0:pos_12] + '1' + pairBin[i][j][pos_3][pos_4][pos_12+1:]
    pair = np.array([])
    for i in range(len(pairBin)):
        for j in range(len(pairBin[0])):
            for k in range(len(pairBin[0][0])):
                for l in range(len(pairBin[0][0][0])):
                    Decimal = ToDecimal(pairBin[i][j][k][l])
                    pair = np.append(pair, Decimal, axis=None)
    parent = np.reshape(pair, (10, 3, 4))
    return parent

# KMeans 根據突變結果計算分類正確率，若未達標從 Fitness 繼續下一輪
def KMeans(k, X, Y, parent):
    # 歐式距離計算
    def Euclidean(X, k, centroid):
        minDistance = np.array([])
        for x in range(len(X)):
            distanceList = np.array([])
            for i in range(k):
                try:
                    distance = np.sqrt(sum(np.square(centroid.get(i) - X[x])))
                except TypeError:
                    continue
                else:
                    distanceList = np.append(distanceList, distance)
            minDistance = np.append(minDistance, np.argmin(distanceList)).astype(int)
        
        # 根據離質心距離最近的分類在一起，並組成 dataframe
        df = pd.DataFrame(zip(X, minDistance, Y))
        df.rename(columns = {0: 'DataX', 1: 'yPlum', 2: 'Y'}, inplace = True)
        return df
    
    def UpdateCentroid(df, X, Y, k):
        # Group 組成來源檢視
        group = df.groupby("yPlum")
        modeNoList = np.array([])
        centroid = {}
        for i in range(k):
            try:
                yLabel = group.get_group(i)
            # print('yLabel', yLabel)
            except KeyError:
                # print("\nKeyError", i, "類")
                continue
            else:
                y = stats.mode(yLabel["Y"])[0][0] # stats.mode 傳出的資料為陣列
                newLabel = Y[yLabel.index]
                modeNo = newLabel.tolist().count(y)
                modeNoList = np.append(modeNoList, modeNo, axis=None)
                poolAccuracy =(modeNo / len(yLabel))
                Xcentroid = sum(yLabel['DataX'])/len(yLabel)
                centroid[i] = Xcentroid
            # print('第 {} 來自屬於 {}: 集區正確率: {}'.format(i, y, poolAccuracy))
        
        Accuracy = (sum(modeNoList) / len(X))
        # print('\n整體正確率: {}'.format(Accuracy))
        # print(centroid)
        return centroid, Accuracy
    
    Accuracy_list = np.array([])
    for i in range(len(parent)):
        centroid = {}
        
        for j in range(len(parent[0])):
            centroid[j] = parent[i][j] # 這裡idx修改
        # 這裡要進行 while ，做 kmeans 搜索
        Accuracy = np.array([])
        w = 0
        # while True: 有 Error 未解決 (while / for choose one)
        for t in range(10):
            df = Euclidean(X, k, centroid)
            centroid, accuracy = UpdateCentroid(df, X, Y, k)
            Accuracy = np.append(Accuracy, accuracy, axis=None)
        Accuracy_list = np.append(Accuracy_list, max(Accuracy))
        # print('\n{}: 整體正確率\n{}'.format(i, Accuracy))
        
    
    print("\nAccuracy List: ", Accuracy_list)
    # print("\nparent", parent)
    return Accuracy_list, parent




# KMeans = KMeans(k, X, Y)
parent = Parent(X)
for i in range(100): 
    fitness = Fitness(parent, X)
    pair = ParentPair(fitness, parent)
    pairBin = Crossover(pair, crossoverRate)
    parent = Mutation(pairBin, mutationRate)
    Accuracy_list, parent = KMeans(k, X, Y, parent)

