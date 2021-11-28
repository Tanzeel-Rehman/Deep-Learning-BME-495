# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 01:20:36 2018

@author: Tanzeel
"""

import numpy as np
from math import exp

class NeuralNetwork:
    def __init__(self,*args, dtype=int):
        self.__network={}
        self.__a={}     # Dictionary to store the outputs of all the layers (Intermediate dictionary)
        self.__dE_dTheta={} #Dictionary to store the partial derivatives for every layer
        for i in range (1, len(args)):
            #np.random.seed(1)
            layers = np.random.normal(0.00,1.0,(args[i-1]+1,args[i]))
            #layers=layers.transpose()
            self.__network['W%i'%i]=layers
            #print(self.__network)
        
    def getLayer(self,layer, dtype=int):
        layer=layer+1
        return (self.__network['W%i'%layer])
    
    
    # More efficient implwmwntation of 1D forward pass
    def forward(self,input_Tensor):
        self.__input_Tensor=input_Tensor
        for layer in range(len(self.__network)):
            layer=layer+1
            weights=self.__network['W%i' %layer]
            weights=weights.transpose()
            output=[]
            tot=np.zeros(weights.shape[0])
            input_Tensor=np.insert(input_Tensor,0,1)
            assert weights.shape[1]==input_Tensor.shape[0], "Incorrect input Tensor. Tensor height should be similar to the input dimensions used while intializing the Net."
            for x in range(weights.shape[0]):
                tot[x]=np.dot(weights[x], input_Tensor)
                output.append(1.0/(1.0+exp(-tot[x]))) 
            input_Tensor=np.delete(input_Tensor,0,0)
            self.__a['W%i' %(layer-1)]=input_Tensor    #Add input tensor to the output dictionary
            input_Tensor=output
            input_Tensor=np.asarray(input_Tensor)
            output=np.asarray(output)
            #output=np.insert(output,0,1)        # Append the first row of output to accomodate the bias
            self.__a['W%i' %layer]=output   #Add all the outputs to dictionary
            #print(input_Tensor)
        return output
    
    # More efficient implementation of2D forward pass
    def forward2D(self,input_Tensor):
        for layer in self.__network:
            weights=self.__network[layer]
            weights=weights.transpose()
            #output=np.zeros_like(input_Tensor)
            output=[]
            tot=np.zeros((weights.shape[0],input_Tensor.shape[1]))
            b=np.ones(input_Tensor.shape[1])
            input_Tensor=np.vstack((b,input_Tensor))
            assert weights.shape[1]==input_Tensor.shape[0], "Incorrect input Tensor. Tensor height should be similar to the input dimensions used while intializing the Net."
            for x in range(weights.shape[0]):
                tot[x]=np.dot(weights[x], input_Tensor)
            for a in range (tot.shape[0]):
                for b in range(tot.shape[1]):
                    output.append(1.0/(1.0+exp(-tot[a,b]))) 
            input_Tensor=output
            input_Tensor=np.asarray(input_Tensor)
            input_Tensor=input_Tensor.reshape(tot.shape[0],tot.shape[1])
            output=np.asarray(output)
            output=output.reshape(tot.shape[0],tot.shape[1])
        return output
    
    # Backpropagation pass single target 
    def backward(self, target):
       for x in reversed (range(len(self.__network))):
           x=x+1
           # Compute the partial derivatives for the output layer
           if x==(len(self.__a)-1):             # Length of network is 2, however the length of output is 3 therefore we need 1 less 
               outputs=self.__a['W%i'%x]
               deltas=(-1*(target-outputs))*outputs*(1-outputs)
               deltas=deltas.reshape((1,deltas.shape[0]))
               lastlayerout=self.__a['W%i'%(x-1)]           # Change the output to previous layer
               lastlayerout=np.insert(lastlayerout,0,1.0)   # Add the 1 as input for the bias term 
               lastlayerout=lastlayerout.reshape((lastlayerout.shape[0],1))
               y=np.dot(lastlayerout,deltas)
               self.__dE_dTheta['W%i'%x]=y
               #print(x)
               #print(self.__dE_dTheta)
           else:  
               # here we have to loop through all the outputs 
               for i in reversed(range(1,len(self.__a))):
                   weights=self.__network['W%i'%i]
                   outputs=self.__a['W%i'%i]
                   deltas=np.zeros_like(weights,dtype=float)
                   delta=(-1*(target-outputs))*outputs*(1-outputs)
                   for a in range(0,weights.shape[0]):
                       for b in range(weights.shape[1]):
                           deltas[a,b]=delta[b]*weights[a,b]
                   deltas=np.sum(deltas,axis=1)
                   #deltas=np.delete(deltas,0,0)
                   deltas=deltas.reshape((1,deltas.shape[0]))
                   Previouslayerout=self.__a['W%i'%(i-1)]   # Change the output to previous layer
                   Previouslayerout=Previouslayerout*(1-Previouslayerout)   #(outh1*(1-outh1))
                   Previouslayerout=Previouslayerout.reshape((Previouslayerout.shape[0],1))
                   y=np.dot(Previouslayerout,deltas)
                   input_Tensor=self.__input_Tensor
                   input_Tensor=np.insert(input_Tensor,0,1) # Add the 1 as input for the bias term 
                   input_Tensor=input_Tensor.reshape(1,input_Tensor.shape[0])
                   a=input_Tensor*y
                   a=a.transpose()
                   self.__dE_dTheta['W%i'%(i-1)]=a
               #print(self.__dE_dTheta)
               #return(self.__dE_dTheta)
                  
   
    def updateParams(self,eta,dtype=float):
        for i in reversed(range(len(self.__network))):
            i=i+1
            weights=self.__network['W%i'%i]
            derivatives=self.__dE_dTheta['W%i'%i]
            new_weights=np.zeros_like(weights)
            for x in range(0,weights.shape[0]):
                new_weights[x]=(weights[x]-eta*derivatives[x])
            self.__network['W%i'%i]=new_weights
        print(self.__network)
        #return(self.__network)
            


