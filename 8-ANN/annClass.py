# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:29:59 2022

@author: abade
"""
import numpy as np 



class NeuralNetwork():
    
    
    def __init__(self, X, y, hidden_units:list, bias=1):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        hidden_units : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.X = X
        self.y = y
        self.y_pred = None
        self.N = self.X.shape[0]
        self.bias = bias
        self.hidden_units = hidden_units
        self.n_in = self.X.shape[1]
        self.n_out = self.y.shape[1]
        self.layers = [self.n_in, *self.hidden_units, self.n_out]
        self.weights = []
        self.h_activation = []
        self.z_activation = []
        self.num_layers = len(self.layers)
        
        

    def initialize_W(self):
        """
        Given an array indicating the number of neurons in each layer 
        initializes the weights randomly multiplied with a scale proportional
        to the number of neurons. 
        It returns the weights TRANSPOSED

        Parameters
        ----------
        Returns
        -------
        weights : list
            List of weights on each layer:
            (n_neurons+1(l), n_neurons(l+1))

        """
        # Substract 1 for indexing
        for i in range(self.num_layers-1):
            bias_neuron = 1
            # Neurons in layer (l) adding the bias term 
            neurons_l = self.layers[i] + bias_neuron
            # Get the scale proportional to the number of hidden units on 
            # layer (l)
            scale = np.sqrt(2/neurons_l)
            # Get the number of neurons on the layer l+1 (without the bias term)
            neurons_l_plus1 = self.layers[i+1]
            # Initialize the weights for each layer on the form of 
            # (n_neurons(l), n_neurons(l+1)).T
            wi = np.random.randn(neurons_l_plus1,neurons_l).T * scale
            self.weights.append(wi)
            


    def softmax(self, y:np.array):
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
    
    
    def forward(self):
        """
        Forward propagation following: 
            z = Σ(x_i*w_ij.T) where i is the neuron in layer l and j is the 
            neuron in tghe layer l+1
        In a Vectorial form:
            z(l+1) = X*w.T(l) where l is the layer we are calculating the 
            activation function 
        Activation function ises the ReLu.
            # 1  if x > 0 
            # 0  otherwise
        Prediction y normalized with the softmax function
            
        Keeps track of: 
            z: Linear combination 'z'activationd function 
            h: Activation function of the linear combination of z
            y_pred: The predicted values for y


        Returns
        -------
        y_hat : TYPE
            DESCRIPTION.
        h_activation : TYPE
            DESCRIPTION.
        z_act : TYPE
            DESCRIPTION.

        """
        
        # Create a column vector for the bias term dimensions (N,1)
        b = np.ones((self.N,1))*self.bias
        h_activation =[]
        z_act = []
   
        # [2, 3, 2]
        #  0  1  2 
        # Iterate thorugh the different layers 
        for i,w in enumerate(self.weights):
            
            # First hidden layer, with values of X*w.T
            if not i:   
                print(f'Hidden layer {i+1}/{self.num_layers-1}. Neurons in layer: {self.layers[i]}')
                # Add a bias term to the X data (N,n_neurons + 1)
                X_tmp = np.column_stack([b, self.X])
                # Append X on the actiovation to use it later 
                self.h_activation.append(X_tmp)
                # Calculate the logistic regression model for layer (l+1):
                    # (N,n_neurons+1(l))*(n_neurons+1(l), n_neurons(l+1))
                    # (N,n_neurons(l+1))
                z_plus1 = X_tmp@w  
                # Append to get track for the backward propagation 
                self.z_activation.append(z_plus1)
                # Apply actiovation function ReLu:
                h_plus1 = np.maximum(0,z_plus1) # (N,n_neurons(l+1))
                # Store the activation function
                self.h_activation.append(h_plus1)
                # h_activation has 2 elements (X, h(1))

            # Outpt layer   
            elif i == self.num_layers-1:
                print(f'Hidden layer {i+1}/{self.num_layers-1}. Neurons in layer: {self.layers[i]}')
                # Obtain activation from layer l
                #   (N,n_neurons(l)) 
                hl = self.h_activation[i]
                # Append the bias term 
                #   (N,n_neurons+1(l)) 
                hl = np.column_stack([b, hl])
                # Calculate the output 
                y_hat = hl@w
                # Nomralize with the softmax function 
                self.y_pred = self.softmax(y_hat)
              
            # Hidden layers
            else:
                # We need to use the activation from layer (l)
                print(f'Hidden layer {i+1}/{self.num_layers-1}. Neurons in layer: {self.layers[i]}')
                hl = self.h_activation[i]
                # Append the bias term 
                hl = np.column_stack([b, hl])  # (N,n_neurons+1(l))
                # Calculate the z linear combination of the next layer
                # (N,n_neurons+1(l))*(n_neurons+1(l), n_neurons(l+1))
                # (N,n_neurons(l+1))
                z_plus1 = hl@w 
                self.z_activation.append(z_plus1)
                # Apply ReLu:
                h_plus1 = np.maximum(0,z_plus1)
                self.h_activation.append(h_plus1)
                

X = np.zeros((10,2))
y = np.zeros((10,2))
hidden_units = [3, 5]
ANN = NeuralNetwork(X, y, hidden_units)

ANN.initialize_W()
ANN.forward()