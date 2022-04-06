# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:21:54 2022

@author: G531
"""

import numpy as np 
import matplotlib.pyplot as plt 
import skimage.io 
from skimage.draw import polygon2mask
import cv2 
import os
import imageio as iio 
import sys
import snake_functions as sf 

#%% Import internal packages 

sys.path.insert(1, r"C:\Users\G531\Documents\1 - Universities\3 - DTU\002 - Course\Semester-02\05 -Advanced Image Analysis\week6\pycode")

#%%  Import video 

path = os.getcwd()
amoeba = iio.get_reader(path + '/data/crawling_amoeba.mov')

#%% 

def rgb2bw(im):
    
    im = im.astype(float)/255
    
    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]
    
    bw = (2*B-(R-G) + 2)/4
    
    return bw
#%% stack images 

for i,im in enumerate(amoeba):
    im = rgb2bw(im)
    if not i:
        I = im
    else:
        I = np.dstack((I,im))
    
#%% Function to change RGB to Black and white 




def computeNormals(C):
    """
    Given a set of points C of a polyedric figure, calculates the norm of the 
    segments AB, being A the point itself and B the points to the right. 
        A = (x_i,y_i)
        B = (x_i+1, y_i+1)
    Once calculates all the norms of the segments AB, then averages such
    segments to calculate the norm of the point P.
        P = (x,y)

    Parameters
    ----------
    C : np.2darray
        A 2D array of dimensions (n,2)

    Returns
    -------
    C_norm : np.2darray
        Returns the norms of the points 

    """
    # MOD: As array has 99, we stack at the end the first value
    C = np.vstack([C, C[0, :]])
    
    # Get points of the point itself (A) and the point to the right (B).
    A = C[:-1, :]
    B = C[1:, :]

    # Compute segments
    AB = B - A

    # Create a vector to store the normals
    AB_norm = np.zeros(AB.shape)
    # Build the norms for each sector
    AB_norm[:, 0] = -AB[:, 1]
    AB_norm[:, 1] = AB[:, 0]

    # rearange so we can compute evarge norms
    L = np.vstack([AB_norm[-1, :], AB_norm[:-1, :]])
    # Compute the means of norm on the segments to calculate the norm on the points
    C_norm = L + AB_norm

    return C_norm


def rigi_elas(X, alpha: int = 0.5, beta: int = 0.5, A_cte: list = [0, 1, -2], B_cte: list = [-1, 4, -6]):

    n = len(X)

    # Identity Matrix
    I = np.eye(n)

    lower = np.ones(2)

    # Diagonal of A Matrix
    A_diag = I*A_cte[-1]
    A_diag[-1, 0] = A_cte[1]
    # Set uper rigth and lower left corners to 0
    A_minus1 = np.diag(np.ones(n-1)*A_cte[1], -1)
    A_minus2 = np.diag(np.ones(n-2)*A_cte[0], -2)
    A_corner = np.diag(lower*A_cte[0], 2-n)
    A = A_diag + A_minus1 + A_minus2 + A_corner

    A += A.T - np.diag(np.diag(A))

    # Diagonal of B Matrix
    B_diag = I*B_cte[-1]
    B_diag[-1, 0] = B_cte[1]
    # Set uper rigth and lower left corners to 0
    B_minus1 = np.diag(np.ones(n-1)*B_cte[1], -1)
    B_minus2 = np.diag(np.ones(n-2)*B_cte[0], -2)
    B_corner = np.diag(lower*B_cte[0], 2-n)
    B = B_diag + B_minus1 + B_minus2 + B_corner

    B += B.T - np.diag(np.diag(B))

    # Construct B matrix which controls riidity and elasticity
    B_int = np.dot(np.linalg.inv(I-alpha*A-beta*B), X)

    return B_int

def snake_expand(C_p: np.array, tau: int, F_ext: np.array, N: np.array):
    """
    Calculates the next step of the snake in the image. 
    The expansion of the snake it's due internal and external forces:
    F_int: 
        Determined by the shape of of the curve. It's calculated with 
        the elasticity and rigidity of the curve. (refer to matrix_B function).
    F_ext:
        Determined by the image. Closely related to two-phase piecewise 
        cte Mumford-Shah model

            F_ext = (m_in-m_out)(2*I-m_in-m_out)

            Where I is the intensity of a pixel (x,y). m_in and m_out are the
            mean intensities of the inside and outside region respectively.

    Then the final expansion can be calculated:
        C = B_int *(C_previous + tau*diag(F_ext)*N_previous)
    Parameters
    ----------
    B : np.2darray
        (n,n) Matrix of internal forces. (refer to matrix_B function).
    C_p : np.array
        Snake points coordinates (n,2) on the previous step. (the ones we want
        to actualize)
    tau : int
        Time step for displacement
    F_ext : np.array
        External Forces (n,)
    N : np.array
        Normals to the points. (refer to computeNormals) (n,2)

    Returns
    -------
    C_next : np.array
        Next step on the snake

    """

    # C = C_p - tau*F_ext*N
    #
    C = rigi_elas(C_p + tau*F_ext*N)
    C = sf.distribute_points(C.T)
    C = sf.remove_intersections(C).T

    return C

def calcFexternal(I:np.array, mu_in:int, mu_out:int):
    
    #F_ext = (mu_in - mu_out)*(I_int - 1/2*(mu_in+mu_out)).reshape(-1, 1)
    F_ext = ((mu_in-mu_out)*(2*I - mu_in - mu_out)).reshape(-1, 1)
    
    return F_ext


def plotsnake(im, C, N, F_ext):

    fig, ax = plt.subplots(figsize=(15, 10))
    C = np.vstack([C, C[0, :]])
    ax.axis('equal')
    ax.plot(C[:, 0], C[:, 1])
    ax.imshow(im, cmap='gray')

    for i in range(C.shape[0]-1):

        x = [C[i, 0], (C[i, 0]-N[i, 0]*F_ext[i,0]*4)]
        y = [C[i, 1], (C[i, 1]-N[i, 1]*F_ext[i,0]*4)]
        
        ax.plot(x, y, 'r')

#%% 

# Set first and last frame
start = 1
stop = 260

# Get 1 image
I1 = I[:,:,0]

# Create circle
n_points = 100
x = np.linspace(0, 2*np.pi, n_points)
r = 60 
row, col = I1.shape
x0, y0 = col/2, row/2
C = np.array([x0 + r*np.cos(x), y0 + r*np.sin(x)]).T
C = np.delete(C, -1, axis=0)

# Show image and circle
fig, ax = plt.subplots()
ax.imshow(I1, cmap='gray')
ax.plot(C[:, 0], C[:, 1], 'r')

# Create mask and get 
M = polygon2mask(im.shape, C[:,::-1])
mu_in = np.mean(I1[M])
mu_out = np.mean(I1[~M])

# Image intensities
idx_x = np.round(C[:, 0]).astype(int)
idx_y = np.round(C[:, 1]).astype(int)

I_int = im[idx_y, idx_x]
# C = np.column_stack([idx_x, idx_y])

# External Force
N = computeNormals(C)
F_ext = calcFexternal(I_int, mu_in, mu_out)
plotsnake(I1, C, N, F_ext)

tau = 10
# Snake Expansion
C = snake_expand(C, tau, F_ext, N)
#%%

t = 10
start = 1
stop=1
for frame in range(start, stop+1):
    Ia = I[:,:,frame]
    print('Frame number: {}'.format(frame))
    for i in range(t):
        print(f'\tIteration {i} mean in: {mu_in}')
        M = polygon2mask(im.shape, C[:,::-1])
        mu_in = np.mean(I1[M])
        mu_out = np.mean(I1[~M])

        # Image intensities
        idx_x = np.round(C[:, 1]).astype(int)
        idx_y = np.round(C[:, 0]).astype(int)
        I_int = im[idx_y, idx_x]
        
        # External Force
        C_int = np.column_stack([idx_x, idx_y])
        N = computeNormals(C)
        F_ext = calcFexternal(I_int, mu_in, mu_out)
        plotsnake(I1, C, N, F_ext)
        
        # Snake Expansion
        C = snake_expand(C, tau, F_ext, N)


#%% 

M = polygon2mask(im.shape, C[:,::-1])
plt.imshow(M)


# TODO: It seems that it's not detecting well the diferences between outside and inside 
# could be that the problem it's when we are converting the image into black 
# and white. we are just using the formula from thhe other cell 
#   - Check what's image ravel
plt.figure()
fig, ax = plt.subplots()
ax.hist(I1.ravel(), bins=256, color = 'k')



    