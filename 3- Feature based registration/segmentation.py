# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 16:29:30 2022

@author: G531
"""

import numpy as np 
import matplotlib.pyplot as plt 
import skimage
import skimage.io
import skimage.feature
import cv2
from itertools import groupby
from operator import itemgetter
import argparse
import os 
import local_features as fl
import random 
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import extract_patches_2d


#%% Features from Gaussian and its derivatives
path = os.getcwd()
# Load the Image 
im = skimage.io.imread(path + '/2labels/training_labels.png').astype(np.float)
# Get the shape f the image 
row, col = im.shape
# Compte the F features 
t = 3
F = fl.get_gauss_feat_im(im,t)
r,c,n = F.shape
n1 = int(np.ceil(np.sqrt(n)))
n2 =int( np.ceil(n/n1))

fig, ax = plt.subplots(n1,n2)
c = 0
for j in range(n1):
    for i in range(n2):
        ax[i][j].imshow(F[:,:,i+j], cmap='jet')
        ax[i][j].set_title(f'layer {c}')        
        c += 1


#%% MULTI FEATURES 
path = os.getcwd()
# Load the Image 
im = skimage.io.imread(path + '/2labels/training_labels.png').astype(np.float)
# Get the shape f the image 
row, col = im.shape
# Compute the F_multi Features
F = fl.get_gauss_feat_multi(im)
rc, t, n = F.shape
# Reshape the F-multi into a (r*c, n*15)
F = F.reshape((rc,t*n))

I = fl.im2col(im) # 9 Rows (patch size) times (columns-2 * rows -2) as the patch moves through the each column and row
# Each column stores the intensity values of the patch 

I = I.reshape((9,row-2,col-2))

fig,ax = plt.subplots(3,3)
for j in range(3):
    for i in range(3):
        ax[i][j].imshow(I[3*i+j], cmap='jet')
        ax[i][j].set_title(f'layer {3*i+j}')



# pt = extract_patches_2d(im, (200,200), max_patches=9)
pt = np.lib.stride_tricks.as_strided(I, shape=())
fig,ax = plt.subplots(3,3)
for j in range(3):
    for i in range(3):
        ax[i][j].imshow(pt[3*i+j], cmap='jet')
        ax[i][j].set_title(f'layer {3*i+j}')

#%% 
path = os.getcwd()
# Load the Image 
im = skimage.io.imread(path + '/2labels/training_labels.png').astype(np.float)


def im2patch(im, npatch):
    
    nx = ny = int(np.floor(np.sqrt(npatch)))
    
    r,c = im.shape
    s0, s1 = im.strides 
    
    nrows =r-nx+1
    ncols = c-ny+1
    shp = nx,ny, nrows,ncols
    strd = s0,s1,s0,s1
    
    out_view = np.lib.stride_tricks.as_strided(im, shape=shp, strides=strd)
    
    return out_view, out_view.reshape(nx*ny,-1)[:,::1]

A, B = im2patch(im,100)

N = A.reshape(A.shape[0]*A.shape[1], A.shape[2], A.shape[3])
fig,ax = plt.subplots(10,10)
for j in range(10):
    for i in range(10):
        ax[i][j].imshow(N[3*i+j], cmap='jet')
        ax[i][j].set_title(f'layer {3*i+j}')
        
        
#%% 

A = np.zeros((10,10))
A[0:5,0:5] = 1
A[5:10, 5:10] = 1
plt.figure()
plt.imshow(A)
# 80, 8
brow, bcol = A.strides

strd = ()
patches = 10

O = np.lib.stride_tricks.as_strided(A, shape=[3,3,], strides=strd)



#%% 

r = 600 
c = 600
a = 100


n1 = np.floor(np.sqrt(a))
n2 = np.floor(a/n1)
print(n1, n2)
print(n1*n2)

xp = np.floor(r/n1)
yp = np.floor(c/n2)
print()



#%%
# features based on image patches
I = I[300:320,400:420] # smaller image such that we can see the difference in layers
pf = im2col(I,[3,3])
pf = pf.reshape((9,I.shape[0]-2,I.shape[1]-2))
        
fig,ax = plt.subplots(3,3)
for j in range(3):
    for i in range(3):
        ax[i][j].imshow(pf[3*i+j], cmap='jet')
        ax[i][j].set_title(f'layer {3*i+j}')
   



























#%% Scale Features 

# Create an array to store the features
F = np.zeros((row,col,z))
t = 10
for i in range(d):
    for j in range(n):
        gx = gauss(t,i)
        gy = gauss(t,j)
        L = cv2.filter2D(cv2.filter2D(im,-1,gx),-1,gy.T)
        F[:,:,i] = L/np.std(L)
    n -= 1

#%% Multi scale Features
# Set variables for the loops 
d = 5 
n = 5
# Specify Number of combinations
z = 15
# Specify an array of scales 
t = [1,2,3]
F = np.zeros((row,col,z*len(t)))
F = np.zeros((row,col,z*len(t)))
for s in t:
    for i in range(d):
        for j in range(n):
            gx = gauss(s,i)
            gy = gauss(s,j)
            L = cv2.filter2D(cv2.filter2D(im,-1,gx),-1,gy.T)
            F[:,:,i+j+s] = L/np.std(L)
        n -= 1
#%%


