# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:14:19 2022

@author: G531
"""

import skimage.io
import matplotlib.pyplot as plt
import maxflow
import numpy as np
from week5 import likelihood, threshold, priorP
from itertools import product

#Image in 16 bytes
I =  skimage.io.imread('../week5/V12_10X_x502.png').astype(np.float)/(2**16-1)

# HISTOGRAM OF THE IMAGE
fig, ax = plt.subplots()
ax.hist(I.ravel(), bins=256, color = 'k')

# histogram 2
plt.figure()
hist, binedg = np.histogram(I*255, bins= 256)
plt.bar(binedg[0:-1], hist, alpha = 0.7)

#%% Likelihood
# By analyzing the histogram we can define 
T1 = 140/255
mu = [90/255, 175/255]

S = threshold(I, T1)

V1,Im = likelihood(I, S, *mu)
plt.figure()
plt.imshow(S, cmap='gray')

#%% 
# Create the graph.
g = maxflow.Graph[float]()

# Add the nodes. nodeids has the identifiers of the nodes in the grid.
# Note that nodeids.shape == img.shape
nodeids = g.add_grid_nodes(I.shape)
beta = 0.1
g.add_grid_edges(nodeids, beta)

# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
w_t = (I - mu[1])**2
w_s = (I - mu[0])**2

g.add_grid_tedges(nodeids, w_t, w_s)

# Find the maximum flow.
g.maxflow()

# Get the segments of the nodes in the grid.
# sgm.shape == nodeids.shape
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
I2 = np.int_(np.logical_not(sgm))



def segmentation_histogram(ax, I, S, edges=None):
    '''
    Histogram for data and each segmentation label.
    '''
    if edges is None:
        edges = np.linspace(I.min(), I.max(), 100)
    ax.hist(I.ravel(), bins=edges, color = 'k')
    centers = 0.5*(edges[:-1] + edges[1:]);
    for k in range(S.max()+1):
        ax.plot(centers, np.histogram(I[S==k].ravel(), edges)[0])


fig, ax = plt.subplots()
ax.imshow(sgm)
ax.set_title('max posterior')

edges = np.linspace(0, 1, 257)

fig, ax = plt.subplots()
segmentation_histogram(ax, I, sgm, edges=edges)
ax.set_aspect(1./ax.get_data_ratio())
ax.set_xlabel('pixel values')
ax.set_ylabel('count')
ax.set_title('segmentation histogram')




# fig, ax = plt.subplots(1,2)
# ax[0].imshow(I, cmap='gray')
# ax[1].imshow(I2, cmap='gray')

# w = I[I2==1]
# b = I[I2==0]

# # HISTOGRAM OF THE IMAGE
# fig, ax = plt.subplots()
# ax.hist(w, bins=256, color = 'r')
# # ax.hist(b, bins=256, color = 'b')