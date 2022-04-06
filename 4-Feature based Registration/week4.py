# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:58:52 2022

@author: G531
"""

import skimage 
import skimage.io 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
from numpy import linalg


#%% IMPLEMENT A FUCTION THAT TAKES P AND Q AND RETURNS THE ROTATION 

# P = np.matrix(np.random.randint(0, 10, size=(2,4)))
P= np.matrix(np.random.uniform(1,10,(2,20)))

R = np.matrix([[2, 4],[5, 6]])
# R = np.eye(2)
t = np.array([[2],[3]])
s = 2

Q = s*R@P + t

fig, ax = plt.subplots()
ax.plot(P[0], P[1], color = 'r', marker='.')
ax.plot(Q[0], Q[1], color = 'b', marker='*')

#%% 

def rotation(P, Q):
    """
    

    Parameters
    ----------
    P : 2d array
        Column Vectors 
    Q : TYPE
        DESCRIPTION.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if P.shape != Q.shape:
        raise TypeError('You must provide the same number of points')
    
    # check if the dimensions are the dimesions thta should have 
    
    # Calculate centroid 
    mu_p = np.mean(P, axis=1) 
    mu_q = np.mean(Q, axis = 1)
    
    # Calculate the scale 
    sq= np.sum(np.sqrt(np.array(np.abs(Q-mu_q)[0])**2 + np.array(np.abs(Q-mu_q)[1])**2))
    sp= np.sum(np.sqrt(np.array(np.abs(P-mu_p)[0])**2 + np.array(np.abs(P-mu_p)[1])**2))
    scale= sq/sp
    # Covariance Matrix 
    C = (Q-mu_q)@(P-mu_p).T
    
    # Apply singular Value decompositon
    U, S, V = linalg.svd(C)
    # U -> The columns are the eigenvectors of C*C.T
    # S -> Singular Values 
    # V -> The rows are the eigen vectors of C.T*C
    
    # Provisional Rotation matrix 
    R_hat = U@V
    # D matrix to deal with reflection 
    D = np.matrix([[1, 0],[0, np.linalg.det(R_hat)]])
    # Rotation Matrix 
    R = R_hat@D
    
 
    t = mu_q - s*R@mu_p
    
    return R, t, s


R1, t1, s1 = rotation(P,Q)




#%%

    
    