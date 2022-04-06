# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:46:00 2022

@author: G531
"""
import numpy as np 
import skimage 
import skimage.io
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import maxflow


#%% OBERVIEW TO MARKOV RANDOM FIELDS

# Heights 
d = np.array([179, 174, 182, 162, 175, 165])

# Estimate gender, F as female, M as Male 
# Likelihood (data) term. Average of males and females 
m = 181 
f = 165
# both foloowing a normal standart  with both equal variance 
# Random Configuration
conf = np.array([m, m, m, m, m, f])

# Likelihood energy 
V1 = int(sum((d-conf)**2))

# Now we have to find to configuration that minimazies the Likelihood energy 
conf_m = np.array([m, m, m, m, m, m])
conf_f = np.array([f, f, f, f, f, f])
F_m = np.array([int(e) for e in (d-conf_m)**2])
F_f = np.array([int(x) for x in (d-conf_f)**2])

# We can obviously see that the conf that minimizes the cost is choose the 
# the elements that are the min comparing both arrays

Con = np.array([m, m, m, f, m, f])
F_op = int(sum((d-Con)**2))

# ADDING THE PRIOR, PENALIZE A LESS-FREQUENT CONFIGURATION (F-M)
# Check the 2-click combinations. For a 1D array only the neighbours in one 
# direction 
u = []
beta = 100 

V2 = d[:-1] != d[1:]

# for i in range(len(d)):
#     if i == len(d)-1:
#         break
#     if Con[i] == Con[i+1]:
#         u.append(0)
#     else:
#         u.append(cost)

# Prior probability
Prior = sum(V2)*beta

# Posterior Energy 
U = V1 + Prior

# In order to find a configuartion that gives a lower Posterior we should have 
# to try all possible configurations. 2^6 = 64 and select the one that 
# minimizes the posterior prob 


#############
# Graph cuts for optimizing MRF
d = np.array([179, 174, 182, 162, 175, 165])
conf = np.array([m, m, m, m, m, f])  # s-node configration
conf_inv = conf = np.array([f, f, f, f, f, m])  # node-t conf
N = len(d) # Numbe of nodes 
# weights
w_s = (d-conf)**2  # s-node weight 
w_t = (d-conf_inv)**2  # node-t weight

# Initialise a graph integer
g = maxflow.GraphInt()

# Ad nodes, we have 6 people, we need 6 nodes. [0,1,2,3,4,5].
nodes = g.add_nodes(N)  # requires an integer and creates a list of range N
# Create edges between nodes: Prior
for i in range(N-1):  # N-1 because there are only 5 edges between the 6 nodes
    g.add_edge(nodes[i], nodes[i+1], beta, beta)
    # specify left node and right node, and the weight of the edge in one 
    # direction and in the other 

# Create the edges between s and nodes and between nodes and t
for i in range(N):  # N because there are one edge for node
    g.add_tedge(nodes[i], w_t[i], w_s[i])

# Run the max flow algorithm
flow = g.maxflow()

# displaying the results
labeling = [g.get_segment(nodes[i]) for i in range(N)]
gend = 'MF'

for i in range(0,N):
    print(f'Person {i} is estimated as {gend[labeling[i]]}') 



######
# Try different values of beta
beta1 = 1000
g1 = maxflow.GraphInt()
nodes = g1.add_nodes(N)

for i in range(N-1):
    g1.add_edge(nodes[i], nodes[i+1], beta1, beta1)
for i in range(N):
    g1.add_tedge(nodes[i], w_t[i], w_s[i])
flow1 = g1.maxflow()
label1 = [g1.get_segment(nodes[i]) for i in range(N)]

for i in range(N):
    print('Person{} is estimates to be {}'.format(i, gend[label1[i]]))


beta2 = 0
g2 = maxflow.GraphInt()
nodes = g2.add_nodes(N)

for i in range(N-1):
    g2.add_edge(nodes[i], nodes[i+1], beta2, beta2)
for i in range(N):
    g2.add_tedge(nodes[i], w_t[i], w_s[i])
flow2 = g2.maxflow()
label2 = [g2.get_segment(nodes[i]) for i in range(N)]

for i in range(N):
    print('Person{} is estimates to be {}'.format(i, gend[label2[i]]))



#%% MARKOV RANDOM FIELDS

# Read Image 
im = skimage.io.imread('../week5/noisy_circles.png').astype(np.float)
Gt = skimage.io.imread('../week5/noise_free_circles.png').astype(np.float)

# Histogram 
hist, binedg = np.histogram(im, bins= 256)
plt.bar(binedg[0:-1], hist)


# Means of the three distributions 
mu_1 = 70 
mu_2 = 130 
mu_3 = 190 

# Smoothness weight 
beta = 100 

# First compute a easy threshold to classify labels and get a ground true seg 
# Not sure if we have to do it with the noisy or the original 
# Select Threshold by inspectig the image 
T1 = 100 
T2 = 150 
S = skimage.io.imread('../week5/noisy_circles.png').astype(np.float)

# Thresholding 
S[S < T1] = 0
S[np.where((S>=T1)&(S<=T2))] = 1
S[S > T2] = 2

#%% Function to create a Threshod 

lp = skimage.io.imread('../week5/noisy_circles.png').astype(np.float)


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

I2= threshold(lp, T1, T2)

    
#%% Markov ranodm Fields: Likelihood 


def likelihood(D,S,*mus):
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
        

L2 = likelihood(im, S, mu_1, mu_2, mu_3)

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


P2 = priorP(S,beta)
T_2 = L2 + P2

#%% Try different Configurations 
####
# Thresholds 3
T1 = 70
T2 = 120

# means for the different values 
mu_1 = 50
mu_2 = 100 
mu_3 = 160

beta = 100 
# First treshold the image 

S3 = threshold(lp, T1, T2)
L3 = likelihood(im, S3, mu_1,mu_2,mu_3)
P3 = priorP(S3, beta)
T3 = L3 + P3

#####
# Thresholds 4
T1 = 120
T2 = 200

# means for the different values 
mu_1 = 80
mu_2 = 150
mu_3 = 220

beta = 100 
# First treshold the image 

S4 = threshold(lp, T1, T2)
L4 = likelihood(im, S4, mu_1,mu_2,mu_3)
P4 = priorP(S4, beta)
T4 = L4 + P4

####
# Thresholds 5
T1 = 95
T2 = 155

# means for the different values 
mu_1 = 65
mu_2 = 125
mu_3 = 185

beta = 100 
# First treshold the image 

S5 = threshold(lp, T1, T2)
L5 = likelihood(im, S5, mu_1,mu_2,mu_3)
P5 = priorP(S5, beta)
T5 = L5 + P5

d = {
     'Likelihood': [L2, L3, L4, L5], 
     'Prior':[P2, P3, P4, P5], 
     'Total':[T_2, T3, T4, T5]
     }


Summary = pd.DataFrame(data=d, index=['Threshold 1','Threshold 2','Threshold 3','Threshold 4'])

print(Summary)

#%% BINARY SEGMENTATION WITH MARKOV RADOM FIELDS
# http://pmneila.github.io/PyMaxflow/tutorial.html#getting-started
 
I = skimage.io.imread('../week5/DTU_noisy.png').astype(float)

# Show

# Histogram 
hist, binedg = np.histogram(I, bins= 256)
plt.bar(binedg[0:-1], hist)

# Divide by 255 so likelihood and prior are not that heavy
I = I/255

# Means
mus = [90/255, 170/255]

# Create the graph.
g = maxflow.Graph[float]()

# Add the nodes. nodeids has the identifiers of the nodes in the grid.
# Note that nodeids.shape == img.shape
nodeids = g.add_grid_nodes(I.shape)

# Add non-terminal edges with the same capacity.
beta = 0.1
g.add_grid_edges(nodeids, beta)

# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
w_t = (I - mus[1])**2
w_s = (I - mus[0])**2

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


#%%
# Obtain where the values of I have been segmented
idx_w = np.where(I2==1)
w = I[idx_w]*255
b = I[I2==0]*255


plt.figure()
hist, binedg = np.histogram(w, bins= 256)
plt.bar(binedg[0:-1], hist, alpha = 0.7)
hist, binedg = np.histogram(b, bins= 256)
plt.bar(binedg[0:-1], hist, color='r', alpha=0.7)
hist, binedg = np.histogram(I*255, bins= 256)
plt.bar(binedg[0:-1], hist, color='black', alpha=0.5)

u, c = np.unique(w, return_counts=1)

