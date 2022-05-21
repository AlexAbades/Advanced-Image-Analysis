# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:12:48 2022

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

#%% Read & Plot
# read the image 
im = skimage.io.imread('test_blob_uniform.png').astype(np.float)
# Create a figure to display it 
fig,ax = plt.subplots(1)
# Show the image 
ax.imshow(im, cmap = 'gray')  # We can change the colo map  

#%% 

# Create a gaussian kernel in a function 

def gauss(sigma, o=0):
    """
    Parameters
    ----------
    sigma : Integer
        Variance of the gaussian
    x : List
        List to which you'd like to apply the gauusian
    o : TYPE, int
        The default is 0.

    Returns
    -------
    g : list
        List of values with the gaussian applied

    """
    s_f = np.ceil(5*np.sqrt(sigma))
    x = np.arange(-s_f, s_f + 1)
    # x = np.linspace(start =-s_f, stop= s_f, num=18)
    print(x)
    # print(x1)
    s = np.sqrt(sigma)
    e = np.exp((-x**2)/(2*s**2))
    if o == 0:
        g = 1/(np.sqrt(2*np.pi*s**2))*e
    elif o == 1:
        g = -x/(s**3*np.sqrt(2*np.pi))*e
    elif o == 2:
        g = (e/(s**3*np.sqrt(2*np.pi)))*(-1+((x**2)/(s**2)))
    elif o == 3:
        g = (-2*(-x/(s**3*np.sqrt(2*np.pi))*e))/s**2 - x/s**2*(e/(s**3*np.sqrt(2*np.pi)))*(-1+((x**2)/(s**2)))
    else:
        print('Please choose between 0, 1, 2 deriatives')
    return np.expand_dims(g,axis=1)
    
t = 3
# Apply gaussian to the vector x, or which is the same create a gaussian kernel 
g = gauss(t)
g1 = gauss(t, 1)
g2 = gauss(t, 2) 
g3 = gauss(t, 3)
# Create a plot for the gaussian 
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True) 
ax.plot(g, label='Gaussian')
ax.plot(g1, label='Gaussian 1st derivative')
ax.plot(g2, label='Gaussian 2nd Derivative')
ax.plot(g3, label='Gaussian 3th Derivative')
ax.legend()

#%% Convolve the image in the x direction with normal Gaussian 

t = 10
g = gauss(t)

# Convolve the image in the x direction with normal Gaussian 
im_f = cv2.filter2D(im,-1,g.T)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im_f, cmap = 'gray')
plt.title('Kernel applied on the X direction', fontsize=30)      

# Convolve the image in the y direcction with Gaussian 
im_f2 = cv2.filter2D(im_f,-1, g.T)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im_f2, cmap = 'gray')
plt.title('Kernel applied on the X and Y direction', fontsize=30)  


# We can do it in one line: first X axis, secondly y axis
#Lg = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T)


#%% Detect BLOBS
# Use the laplacia equation to detect blobs one scale
# If we don't load the image as float, we won't get the same result as the 
# values will remain as integers.
# We will apply first the gaussian filter and then the laplacian in order to 
# smooth the image, reduce the noise and, aditionally, the difference on the 
# pixel's intensity will larger on the center of the figures.

im = skimage.io.imread('test_blob_uniform.png').astype(np.float)
t = 20

def laplacian(I, sigma, smooth=None):
    if smooth:  # Apply smoothness 
        g = gauss(sigma)
        Ix = cv2.filter2D(I,-1,g)
        Iy = cv2.filter2D(I,-1,g.T) 
        # Apply gausians so we can use the semi-group property
        Im = cv2.filter2D(cv2.filter2D(I,-1,g),-1,g.T) 
    else:
        Ix = Iy = I
        Im = I
        
    g2 = gauss(sigma, o = 2)  # 2dn derivative
    I_xx = cv2.filter2D(Ix, -1, g2.T)  # On the X axis we apply the transpose the kernel. As we want to detect the perpendicular edges
    I_yy = cv2.filter2D(Iy, -1, g2)
    lap = sigma*(I_xx + I_yy)
    return lap, Im

# Image applying Laplacian without previous smoothing
I_lap,_ = laplacian(im, t)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
pos = ax.imshow(I_lap, cmap='gray')
fig.colorbar(pos)
plt.title('Laplacian without previous smoothing', fontsize=30)  

# Image Applying kernel with previous smoothing
I_lap,_ = laplacian(im, t, smooth=1)
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
pos = ax.imshow(I_lap, cmap='gray')
fig.colorbar(pos)
plt.title('Laplacian with previous smoothing', fontsize=30)  
#%%
# Once we've applied both kernels, the smoothing and the laplacian, we can find 
# local maxima and local minimum. As we are only intersted on the centers, the 
# highest maximums and the lower minimums we'll apply a threshold. We'll apply 
# find peaks. 

thres = 40

# Get the coordinates from the white circles
c_pos = skimage.feature.peak_local_max(I_lap, min_distance=2, threshold_abs=thres)
# Get the coordinates of the black circles
c_neg = skimage.feature.peak_local_max(-I_lap, min_distance=2, threshold_abs=thres)

# Concat both coordinate arrays in a single array.
coord = np.r_[c_pos, c_neg]

# Plot the centers of the cercles on top the real image.
fig, ax  = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
ax.plot(coord[:,1], coord[:,0], '.r')
plt.imshow(im)

#%% Recap: Create a circle given radius and position

# Set the origin 
O = np.array([2,2])
# Create a array of 360 degrees and convert it into radians 
# (because cos and sin functions)
degrees = np.arange(0,361,1)*np.pi/180
# Set the radius 
r = 1


# Equation circle:
    # (x,y) = r*(cos(alpha), sin(alpha)) + (Ox,Oy)

# Calculate each point of the circle 
circle = r*np.array((np.cos(degrees), np.sin(degrees))).T + O
fig, ax = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
ax.plot(O[0], O[1], '.b', markersize=20, label= 'Origin')
ax.plot(circle[:,0],circle[:,1], '.r', label='Point of a circle')
ax.set_aspect('equal', adjustable='box' )
ax.legend()


#%% DETECT IN SAME SCALE
# We have to create a circle of radius 
t = 325
# Set raduis
r = np.sqrt(2*t)
# set all the angles for a circunference
degrees = np.arange(0,361,1)*np.pi/180
# set the array of all the points of a circle of radius r, in the origin.
circle = r*np.array((np.cos(degrees), np.sin(degrees))).T

# Check one circle
fig, ax  = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
s = coord[0,:] + circle
ax.plot(s[:,1], s[:,0], '.r')
plt.imshow(im)

# We could do a for loop so so that every time gets a cordinate form the list 
# of coordinates and sums it to the array of points of a circle of radius r. 
# Then we store it into a variable 

# fig, ax = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
p = np.array([[],[]]).T
for cor in coord:
    p = np.concatenate((p,circle+cor))

# Plot the circles
fig, ax = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
ax.plot(p[:,1], p[:,0], '.r')
plt.imshow(im)

# Or we can do it spliting each coordinate and do array multiplication 
circX = circle[:,1]
circY = circle[:,0]
coordX = coord[:,1]
coordY = coord[:,0]
# Add a dimension to the array 
circX = np.expand_dims(circX, axis=1)
circY = np.expand_dims(circY, axis=1)
coordX = np.expand_dims(coordX, axis=1)
coordY = np.expand_dims(coordY, axis=1)
# Multiply the array of circle with the radius length and an array of 
# length(number of centers) so we can sum both arrays afterwards
circ_X = circX*np.ones((1,len(coordX)))
circ_Y = circY*np.ones((1,len(coordX)))
# Sum the coordenates of the origin 
circ_X = circ_X + coordX.T
circ_Y = circ_Y + coordY.T
# Plot the circles 
fig, ax = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
ax.plot(circ_X, circ_Y, '.r')
plt.imshow(im)


#%% Detecting Blobs in fidderent scales 

I = skimage.io.imread('test_blob_varying.png').astype(np.float)
Ir = skimage.io.imread('test_blob_varying.png').astype(np.float)
plt.imshow(I)
r,c = I.shape


t = 15 
# Number of scales we ara going to use 
n = 100

# Create an array to store all the images with the Kernel applied 
blob = np.zeros((r,c,n))
scale = np.zeros((n))

for i in range(n):
    # get the number of times we apply a gausian (semi-group property)
    scale[i] = t*i
    Il, Im = laplacian(I,t,smooth = 1)
    blob[:,:,i] = i*Il
    I = Im 

# If a 3 dimensional array is evaluated with Peak_local_max, it returns a 3th 
# column which gives the position in the 3th dimesion 
thres = 40
coord_white = skimage.feature.peak_local_max(blob, min_distance=2, threshold_abs=thres)
coord_black = skimage.feature.peak_local_max(-blob, min_distance=2, threshold_abs=thres)
coord = np.r_[coord_white, coord_black]

# Select the scales we we have use ir for detecting the circles 
s = scale[coord[:,2]]
r = np.sqrt(2*s)

circle = np.array((np.cos(degrees), np.sin(degrees))).T
# Separate coordinates
circX = circle[:,1]
circY = circle[:,0]
coordX = coord[:,1]
coordY = coord[:,0]
# Add a dimension to the array 
circX = r*np.expand_dims(circX, axis=1)
circY = r*np.expand_dims(circY, axis=1)
coordX = np.expand_dims(coordX, axis=1)
coordY = np.expand_dims(coordY, axis=1)
# Multiply the array of circle with the radius length and an array of 
# length(number of centers) so we can sum both arrays afterwards
circ_X = circX*np.ones((1,len(coordX)))
circ_Y = circY*np.ones((1,len(coordX)))
# Sum the coordenates of the origin 
circ_X = circ_X + coordX.T
circ_Y = circ_Y + coordY.T
# Plot 
fig, ax = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
ax.plot(circ_X, circ_Y, '.r')
plt.imshow(Ir)

#%% Function to detect blob in multiple scales 
def centers(I,sig,thres=40,t:list=None, n:int=100):
    
    r,c = I.shape
    blob = np.zeros((r,c,n))
    scale = np.zeros((n))
    
    for i in range(n):
        scale[i] = t*i
        Il, Im = laplacian(I,t,smooth = 1)
        blob[:,:,i] = i*Il
        I = Im

    coord_white = skimage.feature.peak_local_max(blob, min_distance=2, threshold_abs=thres)
    coord_black = skimage.feature.peak_local_max(-blob, min_distance=2, threshold_abs=thres)
    coord = np.r_[coord_white, coord_black]
    
    # Get scales and create radius of the detected centers
    s = scale[coord[:,2]]
    r = np.sqrt(2*s)
    degrees = np.arange(0,361,1)*np.pi/180
    circle = np.array((np.cos(degrees), np.sin(degrees))).T
    coordX = np.expand_dims(coord[:,1], axis=1)
    coordY = np.expand_dims(coord[:,0], axis=1)
    circX = r*np.expand_dims(circle[:,1], axis=1)*np.ones((1,len(coordX)))
    circY = r*np.expand_dims(circle[:,0], axis=1)*np.ones((1,len(coordY)))
    
    cx = circX + coordX.T
    cy = circY + coordY.T    
   
    
    return cx, cy

#%% Apply to real data 

cells = skimage.io.imread('SEM.PNG').astype(np.float)
# Select a portion of the image 
cells = cells[200:500,200:500]
# Plot
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(cells, cmap='gray')


thres = 30   
row,col = cells.shape
# Array of the desired raduis of the circles 
r = np.arange(10, 24.5, step = 0.4)/2
# Convert into sigma, or equivelent notation t. t = 1/2*(r**2)
t = (r**2)*np.sqrt(0.5)  # Why are we squerting the 0.5??
n = len(t)
blob = np.zeros((row,col,n))


for e,i in zip(t,range(n)):
    Il, _ = laplacian(cells,e,smooth = 1)
    blob[:,:,i] = Il


# We are interested only on the negative 
coord = skimage.feature.peak_local_max(-blob, threshold_abs=thres)
t = t[coord[:,2]]
r = np.sqrt(2*t)

degrees = np.arange(0,361,1)*np.pi/180
circle = np.array((np.cos(degrees), np.sin(degrees))).T
coordX = np.expand_dims(coord[:,1], axis=1)
coordY = np.expand_dims(coord[:,0], axis=1)
circX = r*np.expand_dims(circle[:,1], axis=1)*np.ones((1,len(coordX)))
circY = r*np.expand_dims(circle[:,0], axis=1)*np.ones((1,len(coordY)))

cx = circX + coordX.T
cy = circY + coordY.T  

fig, ax = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
ax.plot(cx, cy, '.r')
plt.imshow(cells)


def circles(coord, t):
    
    t = t[coord[:,2]]
    r = np.sqrt(2*t)
    degrees = np.arange(0,361,1)*np.pi/180
    circle = np.array((np.cos(degrees), np.sin(degrees))).T
    
    coordX = np.expand_dims(coord[:,1], axis=1)
    coordY = np.expand_dims(coord[:,0], axis=1)
    circX = r*np.expand_dims(circle[:,1], axis=1)*np.ones((1,len(coordX)))
    circY = r*np.expand_dims(circle[:,0], axis=1)*np.ones((1,len(coordY)))

    cx = circX + coordX.T
    cy = circY + coordY.T
    return cx, cy

r = np.arange(10, 24.5, step = 0.4)/2
# Convert into sigma, or equivelent notation t. t = 1/2*(r**2)
t = (r**2)*np.sqrt(0.5) 
coord = skimage.feature.peak_local_max(-blob, threshold_abs=thres)

cx,cy = circles(coord, t)
fig, ax = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey = True)
ax.plot(cx, cy, '.r',cmap='gray')
plt.imshow(cells)


#%% 














































