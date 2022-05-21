# -*- coding: utf-8 -*-
"""
Created on Sat May 21 16:15:42 2022

@author: G531
"""
import skimage 
import numpy as np 
import skimage 
import skimage.io
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import maxflow


lp = skimage.io.imread('circly.png').astype(np.float)

#%% Function to create a Threshod 

mu_1 = 70  
mu_2 = 120 
mu_3 = 180 


I = np.ones(lp.shape)
I_1 = I*mu_1
I_2 = I*mu_2
I_3 = I*mu_3
beta = 100 

I1 = (lp-I_1)**2
I2 = (lp-I_2)**2
I3 = (lp-I_3)**2

If = np.array([I1, I2, I3])
If = np.argmin(If, axis = 0 )

plt.imshow(If, cmap='gray')


#%% 





def threshold(I, *th):
    """
    Function that thresholds an image given a serie of thresholds. 

    Parameters
    ----------
    I : np.array
        Image.
    *th : Integers 
        Thresholds to apply to the image.

    Returns
    -------
    I_temp: np.array
        A compy of the Image.

    """
    n = len(th)
    
    I_temp = np.zeros(I.shape)
    
    for i in range(n+1):
        
        if i == 0:
            I_temp[I<th[i]] = i
        elif i == n:
            I_temp[I>th[i-1]] = i
        else:
            I_temp[np.where((I>=th[i-1])&(I<=th[i]))] = i  
    return I_temp

# I2= threshold(lp, T1, T2)

    
#%% Markov ranodm Fields: Likelihood 


def likelihood(D, S, *mus):
    """
    Using Markov Random fields we calclate the likelihood as one clique
    potentials. 
    We calculate the likelihood as the squared difference from each pixel 
    to the mean. That means, if we have a initial configuration where class 1
    has a value of 170, and the pixel we are analysing belongs to that class, 
    the one clique potential for that pixel will be 170 - value of that pixel. 
    Value of a pixel it's its intensity.
    V_1(f_i) = (mu(f_i)-d_i)^2
    U(d|f) = SUM(V_1(f_i))
    
    Parameters
    ----------
    D : ndarray
        Image we want to threshold
    S : ndarray
        Site image, labeled image depending on threshold. 
    *mus : Tuple
        Mean values for the labels of the site Image. Have to be in labeling 
        order. If Site image has 3 differet labels, we need 3 different mus. 
        where mu_0 is queal to the value of the label 0 on the site image

    Returns
    -------
    V1 : Scalar
        Likelihood Energy

    """
    # check that we have euqal mus as labels 
    if len(np.unique(S)) != len(mus):
        print('Same number of mus must be given as number of labels in Site image')
        print('Labels of site Image: ', np.unique(S))
        print('Mus provided: ', mus)
        return
    
    # To avoid coressed reference:
    I = np.zeros(S.shape)
    
    for mu,i in zip(mus, range(len(mus))):
        I[S==i] = mu
    
    Dif = (D-I)**2
    V1 = sum(sum(Dif))
    
    return V1, Dif
        

L2, diff = likelihood(lp, If, mu_1, mu_2, mu_3)

#%% Markov Ranodm Fields: Prior

def priorP(S: np.array, beta:int):
    """
    We define the 2 clique potentials for discrete labels which penalizes 
    neighbouring labels being different. With a 4 Neighbour configuration, (+)
    
    V_2(f_i, f_i') = 0 if (f_i = f_i'); beta otherwise
    U(f) = SUM(V2(f_i,f_i')) for i~i'
        
    Parameters
    ----------
    S : np.array
        Site image, labeled image depending on threshold.
    beta : int
        Smothness weight .

    Returns
    -------
    V2: int
        Prior Potential.

    """
    
    # Check if the pixel it's the same as the neighbour to the right and left 
    S_col = S[1:,:]!=S[:-1,:]
    # Check if the pixel it's the same as the upper and lower neighbour
    S_row = S[:,1:]!=S[:,:-1]
    
    # Total amount of different pixels
    total_different = S_col.sum() + S_row.sum()
    V2 = beta * total_different
    
    return V2


P2 = priorP(If,beta)
T_2 = L2 + P2



#%% 

I = skimage.io.imread('bony.png').astype(np.float)

# Means
mus = [130, 190]

# Create the graph.
g = maxflow.Graph[float]()

# Add the nodes. nodeids has the identifiers of the nodes in the grid.
# Note that nodeids.shape == img.shape
nodeids = g.add_grid_nodes(I.shape)

# Add non-terminal edges with the same capacity.
beta = 3000
g.add_grid_edges(nodeids, beta)

# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
w_s = (I - mus[1])**2
w_t = (I - mus[0])**2

g.add_grid_tedges(nodeids, w_t, w_s)

# Find the maximum flow.
g.maxflow()

# Get the segments of the nodes in the grid.
# sgm.shape == nodeids.shape
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
I2 = np.int_(np.logical_not(sgm))


fig, ax = plt.subplots(1,2)
ax[0].imshow(I, cmap='gray')
ax[1].imshow(I2, cmap='gray')


np.sum(I2)


#%% 



I = skimage.io.imread('frame.png').astype(np.float)/255

ceneter = (75, 100)
mask = np.zeros(I.shape)

mask[75-40:75+40, 100-40:100+40] = 1
mask = mask.astype('int')
idx_in = np.where(mask==1)
idx_out = np.where(mask==0)


mu_in = np.mean(I[idx_in])
mu_out = np.mean(I[idx_out])


F_ext = (mu_in - mu_out)*(2*I[75+40, 100-40]-mu_in - mu_out)
