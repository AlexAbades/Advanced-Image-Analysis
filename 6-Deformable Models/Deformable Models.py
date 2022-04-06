# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:02:13 2022

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

# %% Load Data

path = os.getcwd()

im = skimage.io.imread(path + '/data/plusplus.png').astype(float)[:, :, 0]/255
amoeba = iio.get_reader(path + '/data/crawling_amoeba.mov')


# %% Import external packages

sys.path.insert(1, r"C:\Users\G531\Documents\1 - Universities\3 - DTU\002 - Course\Semester-02\05 -Advanced Image Analysis\week6\pycode")
import snake_functions as sf

# %% Simple Approach with plusplus image

# Create Circle

# Get dimensions of the image
col, row = im.shape

# Number of points we want our circle to have
n = 100
# Create an array of n points in 360 degrees
x = np.linspace(0, 2*np.pi, n)
# Set origin of the circle
x0, y0 = col/2, row/2
# Radius of the circle
r = 50
# create the coordinates of the circle
C = np.array([x0 + r*np.cos(x), y0 + r*np.sin(x)]).T
C = np.delete(C, -1, axis=0)
# Get the intensities, remember, (col, row) in images
C_int = C.astype(int)  # Gets the integer part


# Show image and circle
fig, ax = plt.subplots()
ax.imshow(im, cmap='gray')
ax.plot(C_int[:, 1], C_int[:, 0], 'r')


# %%

# Create a mask of the circle so we can substract the inner mean and outer mean
mask = polygon2mask(im.shape, C)

# Inside mean
mu_in = np.mean(im[mask])
# Outside mean
mu_out = np.mean(im[~mask])

# Visualize the intensities on the circle
idx_x = np.round(C[:, 0]).astype(int)
idx_y = np.round(C[:, 1]).astype(int)
# Image intensities
I_int = im[idx_y, idx_x]

# Plot
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(range(1, 100), I_int)
ax.axhline(y=mu_in, color='black', linestyle='-', alpha=0.5)
ax.axhline(y=mu_out, color='black', linestyle='-', alpha=0.5)
ax.axhline(y=(mu_in+mu_out)/2, color='r', linestyle='-')

# %%  Compute magnitude of snake displacement

# Basically, if the point has lowervalue than outside mean moves towards the
#  inside, if the point has mean higher than mean in, moves towrads outside.

def calcFexternal(I:np.array, mu_in:int, mu_out:int):
    
    # F_ext = (mu_in - mu_out)*(I_int - 1/2*(mu_in+mu_out)).reshape(-1, 1)
    F_ext = ((mu_in-mu_out)*(2*I - mu_in - mu_out)).reshape(-1, 1)
    
    return F_ext


# BUG: If I use The other equation (2*I..) it crashes because it says the 
# snake it's out of boundries 
    

F_ext = calcFexternal(I_int, mu_in, mu_out)

# %% Calculate the normals of points
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

# %% Display the norms of each point on the circle


norm = computeNormals(C)


    

def plotsnake(im, C, F_ext, N):

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('equal')
    C = np.vstack([C, C[0, :]])
    ax.plot(C[:, 0], C[:, 1])
    ax.imshow(im, cmap='gray')

    for i in range(C.shape[0]-1):
        # MODI: 
        y = [C[i, 1], (C[i, 1]-N[i, 1]*F_ext[i])[0]]
        x = [C[i, 0], (C[i, 0]-N[i, 0]*F_ext[i])[0]]

        ax.plot(x, y, 'r')


plotsnake(im, C, F_ext, norm)


# %% Compute B


def rigi_elas(X, alpha: int = 0.5, 
                 beta: int = 0.5, 
                 A_cte: list = [0, 1, -2], 
                 B_cte: list = [-1, 4, -6]):

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


# %%

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
    # BUG: The formula says +, but if I sum, the snake shrinks instead of expand. 
    # Problem with some sign
    # C = rigi_elas(C_p + tau*F_ext*N)
    C = rigi_elas(C_p - tau*F_ext*N)
    # C_old = C.copy()
    C = sf.distribute_points(C.T)
    # Delta_C = np.max(np.abs(C - C_old))
    # print(Delta_C)
    C = sf.remove_intersections(C).T
    
    return C


# %%
tau = 10
C = snake_expand(C, tau, F_ext, norm)
C = sf.distribute_points(C)


# Visualize the intensities on the circle
idx_x = np.round(C[:, 0]).astype(int)
idx_y = np.round(C[:, 1]).astype(int)
# Image intensities
I_int = im[idx_y, idx_x]

F_ext = (mu_in - mu_out)*(I_int - 1/2*(mu_in+mu_out)).reshape(-1, 1)
N = computeNormals(C)
plotsnake(im, C, F_ext, norm)

# %%
m2 = polygon2mask(im.shape, C[:, ::-1])
plt.figure()
plt.imshow(m2)


# %%



# %%



im = skimage.io.imread(path + '/data/plusplus.png').astype(float)[:, :, 0]/255



num_iter = 25
tau = 2
# Get dimensions of the image
col, row = im.shape

# Number of points we want our circle to have
n = 100
# Create an array of n points in 360 degrees
x = np.linspace(0, 2*np.pi, n)
# Set origin of the circle
x0, y0 = col/2+25, row/2-50
# Radius of the circle
r = 70

# Create the coordinates of the circle
C = np.array([x0 + r*np.cos(x), y0 + r*np.sin(x)]).T
# Delete the last entry (angle 360º = 0º)
C = np.delete(C, -1, axis=0)

# Get the integers of the circle location
idx_x = np.round(C[:, 0]).astype(int)
idx_y = np.round(C[:, 1]).astype(int)

# Create a mask of the circle so we can substract the inner mean and outer mean
mask = polygon2mask(im.shape, C[:, ::-1])

# Inside mean
mu_in = np.mean(im[mask])
# Outside mean
mu_out = np.mean(im[~mask])

# Image intensities
I_int = im[idx_y, idx_x]

# Calculate external Forces and reshaped to (n,1)
# MOD
# F_ext = (mu_in - mu_out)*(I_int - 1/2*(mu_in+mu_out)).reshape(-1, 1)
F_ext = calcFexternal(I_int, mu_in, mu_out)

# Compute the normals of each point (direction in which isgoing to move)
N = computeNormals(C)

plotsnake(im, C, F_ext, N)


for i in range(num_iter):
    # if i > 10:
    #     tau = 2
    # if i > 20:
    #     tau = 1
    # if i > 30:
    #     tau = 0.8
    # if i > 40:
    #     tau > 0.5
    # Calculate the next step of the snake
    C = snake_expand(C, tau, F_ext, N)
    # Apply costrains
    C = sf.distribute_points(C)
    C = sf.remove_intersections(C.T).T
    # Create a new mask
    mask = polygon2mask(im.shape, C[:, ::-1])
    # Calculate Means:
    # Inside mean
    mu_in = np.mean(im[mask])
    print('Iteration: {}, mean inside = {}'.format(i, mu_in))
    # Outside mean
    mu_out = np.mean(im[~mask])
    # Get the integers of the circle location
    idx_x = np.round(C[:, 0]).astype(int)
    idx_y = np.round(C[:, 1]).astype(int)
    # Image intensities
    I_int = im[idx_y, idx_x]
    # Calculate external Forces and reshaped to (n,1)
    # MOD
    # F_ext = (mu_in - mu_out)*(I_int - 1/2*(mu_in+mu_out)).reshape(-1, 1)
    F_ext = calcFexternal(I_int, mu_in, mu_out)
    # Compute the normals of each point (direction in which isgoing to move)
    N = computeNormals(C)
    plotsnake(im, C, F_ext, N)



#%% 

def implicit_smoothing(a, b, X):
    n = len(X)

    A = np.diag(np.full(n, -2)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1) + np.diag(np.ones(1), n-1) + np.diag(np.ones(1), -n+1)
    B = np.diag(np.full(n, -6)) + np.diag(np.full(n-1, 4), 1) + np.diag(np.full(n-1, 4), -1) + np.diag(np.full(1, 4), n-1) + np.diag(np.full(1, 4), -n+1) + \
        np.diag(np.full(n-2, -1), 2) + np.diag(np.full(n-2, -1), -2) + np.diag(np.full(2, -1), n-2) + np.diag(np.full(2, -1), -n+2) 

    return np.dot(np.linalg.inv(np.identity(n) - a * A - b * B), X)
