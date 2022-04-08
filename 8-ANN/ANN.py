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


# Output:
#   X - 2n x 2 array of points (there are n points in each class)
#   T - 2n x 2 target values
#   x - regular sampled points on the area covered by the points that will
#       be used for testing the neural network
#   dim - dimensionality of the area covered by the points
    
    
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


laye = neurons(2, 3, 2)
#%% Weights 
def initialize_W(layers):
    # Return the transoped weights w.T
    weights = []
    
    for l in range(len(layers)-1):
        bias = 1
        n = layers[l] + bias
        scale = np.sqrt(2/n)
        wi = np.random.randn(layers[l+1],n).T * scale
        weights.append(wi)
        
    return weights
      

w3 = initialize_W(laye)


#%% SoftMax function 

def softmax(y:np.array):
    """
    Softmax function. 
        y = e^y/Σ(e^y)  for i = 0 to K being K number of classes.
    Nomralizes the y values onto a range of ∈ [0,1]
    
    Parameters
    ----------
    y : np.array
        Array of values, dimensions of (n,K) being n the number of points 
        and K the number of classes 

    Returns
    -------
    Y_norm : np.array
        Array of values normalized 

    """
    # Create the exponential on all 2d array 
    Y_exp = np.exp(y)
    # Sum on the x direction
    total = np.sum(Y_exp, axis=1).reshape(-1,1)
    # Normalize 
    Y_norm = np.round(Y_exp/total)
    
    return Y_norm


#%% Forward 


        
def forward(X, weights):
    
    N = X.shape[0]
    bias = np.ones((N,1)) 
    h_activation =[]
    z_act = []
    
    for i,w in enumerate(weights):
        
        if not i:   
            # First hidden layer, with values of X
            print('First hidden layer', i)
            X_tmp = np.column_stack([bias, X])
            h_activation.append(X_tmp)
            z = X_tmp@w  # (1,3)
            z_act.append(z)
            h = np.maximum(0,z) # It works with 2d arrays 
            # h_activation.append(h)

            
        elif i == (len(weights) - 1):
            # Outpt layer 
            # Append a one to the prevous activation layer 
            h = np.column_stack([bias, h])
            h_activation.append(h)
            y_hat = h@w
            # Nomralize 
            y_hat = softmax(y_hat)
          
        else:
            # Hidden layers
            print('Hidden layers', i)
            hi = np.column_stack([bias, h])
            z_act = hi@w
            h = np.maximum(0,z)
            h_activation.append(h)

            
            
    return y_hat, h_activation, z_act




    


y1, h1 = forward(X, w3)




#%% Back Propagation 


def backprop(y_pred, y_real, layers, h_act, weights, z_act):
    deltas = []
    Q = []
    # Reverse activation list first element is the last
    h_act = h_act[::-1]
    # last layer
    d = len(layers)-1
    # print(d)
    for i in range(d):
        print(i)
        # Last layer delta
        if not i:
            d_layer = y_pred - y_real
            # We need the last activation
            # print(i, d)
            Q_tmp = h_act[i].T@d_layer
            deltas.append(d_layer)
            Q.append(Q_tmp)
        # other layers
        
        else:
            print('Not the last layer', i)
            # Derivative of the activation in the hidden layer 
            # Get the activation from the layer on the right (l+1) but removing 
            # the bias term
            h_tmp = h_act[i-1][:,1:] # (l{l+1})
            # Calculate the derivative of the activation function 
            a = np.zeros(h_tmp.shape)
            a[h_tmp!= 0] = 1
            # Get the delta from the layer on the right 
            d_next = deltas[i-1]
            # get the weights from the layer on the right 
            w_next = weights[i-1][1:,:]
            # calculate the delta on the layer (partial derivatives respect w)
            d_layer = a * (w_next.T@d_next.T).T
            # Store delta on the layer (it's storing in reverse order )
            deltas.append(d_layer)
            # Calculate the Q 
            Q_tmp = h_act[i].T@d_layer
            Q.append(Q_tmp)
            
        # Reverse order of deltas and Q 
        deltas = deltas[::-1]
        Q = Q[::-1]
        
    return Q, deltas 
    

q,d = backprop(y1, T, laye, h1, w3)


def updateW(lrate, weights, Q):
    new_w = []
    for q,w in zip(Q,weights):
        w_new = w-lrate*q
        new_w.append(w_new)
    
    return new_w



new_w = updateW(0.001, w3, q)


#%%

y2, h2 = forward(X, new_w)










#%% 

laye = neurons(2, 3, 2)

w = initialize_W(laye)

n_iter = 10 

for i in range(n_iter):
    print('Iter: ', i)
    
    y, h = forward(X, w)
    
    Q, d = backprop(y, T, laye, h, w)

    w = updateW(0.001, w, Q)
    print(w[0])
    
    

    