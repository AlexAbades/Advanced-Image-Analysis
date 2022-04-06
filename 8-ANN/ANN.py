# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:55:34 2022

@author: G531
"""



import numpy as np
import matplotlib.pyplot as plt
import os
import sys 
import cv2 
import make_data as mkd

#%% 



n = 1000
example_nr = 2
noise = 1.2

X, T, x, dim = mkd.make_data(example_nr, n, noise)
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(X[0:n,0],X[0:n,1],c = 'red', alpha = 0.3, s = 15)
ax.scatter(X[n:2*n,0],X[n:2*n,1],c = 'green', alpha = 0.3, s = 15)
ax.set_aspect('equal', 'box')
plt.title('training')
fig.show



    
    
#%% Before training, you should make data have zero mean and std of 1
    
c = X.mean(axis = 0)
std = X.std(axis = 0)
x_c = (x - c)/std
X_c = (X - c)/std

fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(X_c[0:n,0],X_c[0:n,1],c = 'red', alpha = 0.3, s = 15)
ax.scatter(X_c[n:2*n,0],X_c[n:2*n,1],c = 'green', alpha = 0.3, s = 15)
ax.set_aspect('equal', 'box')
plt.title('Zero mean training')
fig.show


#%% Simple 3 layer network, 1 hidden layer 

n_in = 2
n_hlayer = 3 
n_out = 2
# scale = 
w_1 = np.random.rand(n_hlayer, n_in+1).T
w_2 = np.random.rand(n_out, n_hlayer+1).T



def neurons(n_in, hlayers:list, m_out):
    
    hlayers = np.array(hlayers)
    # n_in = X.shape[1]
    # n_out = y.shape[1]
    layers = np.insert(hlayers, 0, n_in)
    layers = np.append(layers, n_out)
    return layers 


def initialize_W(layers):
    
    weights = []
    
    for l in range(len(layers)-1):
        bias = 1
        n = layers[l] + bias
        scale = np.sqrt(2/n)
        wi = np.random.randn(layers[l+1],n).T * scale
        print(wi)
        weights.append(wi)
        
    return weights
        
        
def forward(X, weights):
    
    N = X.shape[0]
    bias = np.ones((N,1)) 
    
    for i,w in enumerate(weights):
        print(i)
        if not i:   
            # Input Layer
            print('input layer', i)
            X_tmp = np.column_stack([bias, X])
            z = X_tmp@w  # (1,3)
            h = np.maximum(0,z) # It works with 2d arrays 
            
        elif i == (len(weights) - 1):
            # Outpt
            print('output layer', i)
            h = np.column_stack([bias, h])
            y = h@w
            # Still have to rearrenge the y inot 0-1 range with softmax func
        else:
            # Hidden layers
            print('Hidden layers', i)
            hi = np.column_stack([bias, h])
            z = hi@w
            h = np.maximum(0,z)
    return y

laye = neurons(2, [3, 4], 2)



w3 = initialize_W(laye)


z1 = np.random.randn(1,5)




X1 = X[:2,:]
N = X1.shape[0]
bias = np.ones((N,1))
wi = np.column_stack([bias, X1])


y = forward(X, w3)

