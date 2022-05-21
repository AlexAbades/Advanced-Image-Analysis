# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:24:21 2022

@author: G531
"""

import numpy as np 

def rigi_elas(X, alpha: int = 0.5, beta: int = 0.5, A_cte: list = [0, 1, -2], B_cte: list = [-1, 4, -6], implicit:bool=True):

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

    if implicit:
        # Construct B matrix which controls riidity and elasticity
        B_int = np.dot(np.linalg.inv(I-alpha*A-beta*B), X)
    else:
        B_int = np.dot((I+alpha*A+beta*B), X)
    return B_int