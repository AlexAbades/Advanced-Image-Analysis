# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:15:41 2022

@author: G531
"""

def back_prop(self):
    """
    Back propagation of the ANN.
    Calculates the partial derivatives and and stores the elements to 
    calculate the Q which are used later to update the weights
    
    Partial derivative from Cost function with respect activation function:
        - ∂C/∂a --> d(L) =  y_pred - y_true
        Dimensions: (N, n_neurons(L))
    Partal derivative of L with respect weights:
        - ∂z/∂w --> Q(l) = [1, h(l-1)].T*d(l)  
        Dimensions: (n_neurons+1(L-1), n_neurons(L))
    Partal derivative of z with respect weights (derivative of ReLu):
        - ∂a/∂z --> 1 if z > 0
                    0 if z < 0
        Dimensions: (n_neurons+1(L-1), n_neurons(L))
        
    Returns
    -------
    None.

    """
    b = np.ones((self.N,1))*self.bias
    for i in range(self.num_layers-1):
        print('loop: ',i, 'Neurons in layer :', self.layers[self.num_layers - 1 - i])
        # Last layer 
        #[0]
        if not i:
            # Partial derivative from Cost function with respect activation 
            # function: ∂C/∂a:
                # (N, n_neurons(L))
            d_layer = self.y_pred - self.y_true
            print(self.deltas)
            self.deltas.append(d_layer)
            print('delta ini: ', d_layer)
            # Get the activation from the previous layer:
            h_minus1 = self.h_activation[self.num_layers - 2 - i]
            print('h_activation ')
            print(h_minus1)
            # Append the bias term (N, n_neurons+1(L-1))
            h_minus1 = np.column_stack([b, h_minus1])
            # Partal derivative of L with respect weights:
                # ∂z/∂w 
                # (n_neurons+1(L-1), n_neurons(L))
            # print('h layer: ', h_minus1.shape)
            Q_minus = h_minus1.T@d_layer
            self.Q.append(Q_minus)
            # print(Q)
            # print(Q.shape)
                
            # n_neurons+1(L-1), n_neurons(L)
        #☺ delta [d(L-2),d(L-1),d(L)]    
        #[1, 2, 3]    
        else:
            # delta in the layer (l+1)--> (N, n_neurons(l+1))
            d_plus1 = self.deltas[0]
            # weights in the layer (l+1) --> (n_neurons+1(l), n_neurons(l+1))
            W_plus1 = self.weights[-i]
            # print('W2 with bias')
            # print(W_plus1)
            # Delete the bias term --> (n_neurons(l), n_neurons(l+1))
            # print('W2 without bias')
            W_plus1 = W_plus1[1:,:]
            # print('W as anna')
            # print(self.weights[-1][:-1])
            
            # linear combination WITHOUT actvation function in layer (l)
            z = self.z_activation[-i]
            # Derivative of ReLu: (N, n_neurons(l))
            a_derivative = np.zeros(z.shape)
            a_derivative[z>0] = 1                
            # delta in hidden layers --> (N, n_neurons(l))
            d_layer = a_derivative*(W_plus1@d_plus1.T).T
            # Insert delta at the begining 
            self.deltas.insert(0, d_layer)
            # get the activation on the previuos layer --> (N, n_neurons+1(l-1))
            h_minus = self.h_activation[-1-i]
            # Partal derivative of l with respect weights:
                # ∂z/∂w 
                # (n_neurons+1(L-1), n_neurons(L))
            Q_minus = h_minus.T@d_layer
            # Append Q at the beggining og the list
            self.Q.insert(0, Q_minus)
            # print(Q_minus.shape)
            # 
            # d_plus1 = self.deltas[self.num_layers - 1 - i]
    # Reset the linear combination and the activation function and deltas 
    self.z_activation = []
    self.h_activation = []
    self.deltas = []
    
    
    




def forward_prop(self):
    """
    Forward propagation following: 
        z = Σ(x_i*w_ij.T) where i is the neuron in layer l and j is the 
        neuron in tghe layer l+1
    In a Vectorial form:
        z(l+1) = X*w.T(l) where l is the layer we are calculating the 
        activation function 
    Activation function ises the ReLu.
        # x  if x > 0 
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

    # [2, 3, 2]
    #  0  1  2 
    # Iterate thorugh the different layers 
    for i,w in enumerate(self.weights):
        print(i, '/', self.num_layers)
        # First hidden layer, with values of X*w.T
        if not i:   
            print(f'First layer {i+1}/{self.num_layers-1}. Neurons: {self.layers[i+1]}')
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

        # Outpt layer. 
        elif i == self.num_layers-2:
            print(f'Output layer {i+1}/{self.num_layers-1}. Neurons: {self.layers[i+1]}')
            # Obtain activation from layer l
            #   (N,n_neurons(l)) 
            hl = self.h_activation[i]
            # Append the bias term 
            #   (N,n_neurons+1(l)) 
            hl = np.column_stack([b, hl])
            # print(hl.shape)
            # Calculate the output:
                # (N, n_neurons(L))
            y_hat = hl@w
            # Nomralize with the softmax function 
            self.y_pred = self.softmax(y_hat)
          
        # Hidden layers
        else:
            # We need to use the activation from layer (l)
            print(f'Hidden layer {i+1}/{self.num_layers-1}. Neurons: {self.layers[i+1]}')
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
            
            
            
            
#%% 


#%% 

hidden_units = [3]
ANN = NeuralNetwork(X_c, T, hidden_units)

ANN.initialize_W()
ANN.forward1()





#%% 
ANN.forward_prop()

y1_forwclass = ANN.y_pred

#%% 

ANN.back1()

#%% 


hidden_units = [3]
# Use data normalized 
ANN = NeuralNetwork(X_c, T, hidden_units)
n_iterations = 500
learn_rate=0.001
ANN.trainANN(n_iterations, learn_rate)


y_pred = ANN.y_pred


#%% 







#%%



def NN_MODEL(x,T,n,itera,hid_units,hid_lay):
    
    # #initialize weights
    W= ANN.weights
    # W1 = np.random.randn(3,hid_units)
    # W.append(W1)
    # for e in range(hid_lay):
    #     Wi = np.random.randn(hid_units+1,2)
    #     W.append(Wi)
    print(W)
    #For each iteration
    for i in range(itera):
        print(i)
        z = np.c_[x, np.ones((x.shape[0],1))]@W[0]
        
        for l in range(hid_lay):
            h = np.maximum(z, 0)
            y_hat = np.c_[h, np.ones((x.shape[0],1))]@W[l+1]
        
        y = np.exp(y_hat)/np.sum(np.exp(y_hat), axis=1, keepdims=True)
        
        #loss funtion, prediction - target value
        L2=y-T
        #partial derivative
        dL2=np.c_[h, np.ones((x.shape[0],1))].T@L2
        
        #for the hidden layer, backpropagation
        a=np.zeros(z.shape)
        a[z>0]=1
        L=W[1][:-1]@L2.T
        L1=a*L.T
        dL1=np.c_[x, np.ones((x.shape[0],1))].T@L1
        print('delta 2: ', dL2)
        print('delta 1: ', dL1)
        #Update weights:
        W[0]=W[0]-n*dL1
        W[1]=W[1]-n*dL2
        
    return W,y
    
    
def test_NN(xtest,W_train):
    z = np.c_[xtest, np.ones((xtest.shape[0],1))]@W_train[0]
    h = np.maximum(z, 0)
    y_hat = np.c_[h, np.ones((xtest.shape[0],1))]@W_train[1]
    ytest = np.exp(y_hat)/np.sum(np.exp(y_hat), axis=1, keepdims=True)
 
    return ytest



#%% PARAMETERS 



itera=1 #iterations
n=0.001 #learning rate
hid_units= 3 
hid_lay= 1 


#TRAIN NN
W_train, y_f = NN_MODEL(X_c, T, n, itera,hid_units, hid_lay)



#%% initialize weights
W=ANN.weights

#For each iteration
x = X_c
z = np.c_[x, np.ones((x.shape[0],1))]@W[0]

for l in range(hid_lay):
    h = np.maximum(z, 0)
    print(h.shape)
    y_hat = np.c_[h, np.ones((x.shape[0],1))]@W[l+1]

y_anna = np.exp(y_hat)/np.sum(np.exp(y_hat), axis=1, keepdims=True)