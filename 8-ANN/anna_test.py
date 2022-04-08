# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:59:01 2022

@author: abade
"""
import numpy as np
import matplotlib.pyplot as plt
import make_data




def NN_MODEL(x,T,n,itera,hid_units,hid_lay):
    
    #initialize weights
    W=[]
    W1 = np.random.randn(3,hid_units)
    W.append(W1)
    for e in range(hid_lay):
        Wi = np.random.randn(hid_units+1,2)
        W.append(Wi)
   
    #For each iteration
    for i in range(itera):
        z = np.c_[x, np.ones((x.shape[0],1))]@W[0]
        
        for l in range(hid_lay):
            h = np.maximum(z, 0)
            y_hat = np.c_[h, np.ones((x.shape[0],1))]@W[l+1]
        
        print(y_hat)
        y = np.exp(y_hat)/np.sum(np.exp(y_hat), axis=1, keepdims=True)
        
        #loss funtion, prediction - target value
        L2=y-T
        #partial derivative
        dL2=np.c_[h, np.ones((x.shape[0],1))].T@L2
        
        #for the hidden layer, backpropagation
        a=np.zeros(z.shape)
        a[z>0]=1
        L=W[1][:-1]@L2.T
        L1=a*L.T
        dL1=np.c_[x, np.ones((x.shape[0],1))].T@L1
    
        #Update weights:
        W[0]=W[0]-n*dL1
        W[1]=W[1]-n*dL2
        
    return W,y
    
    
def test_NN(xtest,W_train):
    z = np.c_[xtest, np.ones((xtest.shape[0],1))]@W_train[0]
    h = np.maximum(z, 0)
    y_hat = np.c_[h, np.ones((xtest.shape[0],1))]@W_train[1]
    ytest = np.exp(y_hat)/np.sum(np.exp(y_hat), axis=1, keepdims=True)
 
    return ytest



#%% PARAMETERS 

n_points = 300
example_nr = 2
noise = 1.2
#OBTAIN DATA
X, T, xtest, dim = make_data.make_data(example_nr, n_points, noise)

#Standarize data
c = X.mean(axis = 0)
std = X.std(axis = 0)
xtest_c = (xtest - c)/std
X_c = (X - c)/std


W1 = np.random.randn(3,3)
W2 = np.random.randn(4,2)
W = [W1, W2]

itera=1 #iterations
n=0.001 #learning rate
hid_units= 3#hidden units (flexible structure)
hid_lay= 1#hidden layers (flexible strucutre)


#TRAIN NN
W_train,y_f = NN_MODEL(X_c,T,n,itera,hid_units,hid_lay)


#TEST NN
ytest=test_NN(xtest_c,W_train)
arg=np.argmax(ytest,axis=1)