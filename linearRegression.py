# -*- coding: utf-8 -*-
"""
Created on Sun May 20 10:13:46 2018

@author: ZhenXi
"""
import numpy as np
import matplotlib.pyplot as plt

def featureNormalization(X):
    
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X,axis = 0)
    X_norm = (X-mu) /sigma
    
    return X_norm

def gradientDescent(X,y,theta,learningRate):
    
    theta = theta - 1/np.shape(X)[0]*np.dot(X.T,np.dot(X,theta)-y)*learningRate
    
    return theta

def costCompute(X,y,theta):
    sampleNum = np.shape(X)[0]
    cost = 1/(2*sampleNum)*np.dot((np.dot(X,theta)-y).T,np.dot(X,theta)-y)
    
    return cost

def loadData(fileName):
    
    Data = np.loadtxt(fileName,delimiter = ',')
    featureNum = np.shape(Data)[1] - 1
    print("The number of the features:")
    print(featureNum)
    sampleData = Data[:,0:featureNum]
    sampleResult = Data[:,2]
    
    return sampleData, sampleResult, featureNum


if __name__=="__main__": 
    
    # load data from the file
    # there are two feature in the file. One is the size of houses and the other is the number of bedrooms in houses.
    fileName = 'E:\\MachineLearning\\machine-learning-ex1\\ex1\\ex1data2.txt'
    sampleData, sampleResult, featureNum = loadData(fileName)
    
    # implement feature normalization
    sampleData = featureNormalization(sampleData)
    
    sampleData = np.c_[np.ones(np.shape(sampleData)[0]),sampleData]
    sampleResult = np.reshape(sampleResult,(47,1))
    
    print("The average value of each feature after the featuremalization:")
    print(np.mean(sampleData,axis=0))
    print("After the feature normalization, the means of each feature is approximately 0.")
    print("The standard deviation of each feature after featuremalization:")
    print(np.std(sampleData,axis = 0))
    print("The standard deviation of each feature is 1 after normalizing the features")
    
    
    
    
    
    # indicate the maximal step 
    stepMax = 100
    
    #indicate the learning rate
    rate = 0.03
    
    cost = np.zeros((6,stepMax))
   
    # compute the costs with respect to different learning rates
    for j in range(6):
        theta = np.random.randn(featureNum+1,1)
        rate = rate *2
        for i in range(stepMax):
            cost[j][i] = costCompute(sampleData,sampleResult,theta)
            theta = gradientDescent(sampleData,sampleResult,theta,rate)
    
    # plot 
    plt.subplot(2,1,1)
    plt.plot(np.arange(1,stepMax+1,1),cost[0],label = 'rate = 0.06')
        
    plt.plot(np.arange(1,stepMax+1,1),cost[1],label = 'rate = 0.12')
    plt.plot(np.arange(1,stepMax+1,1),cost[2],label = 'rate = 0.24')
    
    plt.plot(np.arange(1,stepMax+1,1),cost[3],label = 'rate = 0.48')
    plt.plot(np.arange(1,stepMax+1,1),cost[4],label = 'rate = 0.96')
    plt.legend(loc = 'upper right')

    plt.subplot(2,1,2)
    plt.plot(np.arange(1,stepMax+1,1),cost[5],label = 'rate = 1.92')
    plt.legend(loc = 'upper right')