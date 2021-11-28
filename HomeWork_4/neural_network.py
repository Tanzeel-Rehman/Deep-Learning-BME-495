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
            self.__network['W%i'%i] = np.random.normal(0.00,1,(args[i-1]+1,args[i]))
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
            input_Tensor=np.insert(input_Tensor,0,1.0)
            assert weights.shape[1]==input_Tensor.shape[0], "Incorrect input Tensor. Tensor height should be similar to the input dimensions used while intializing the Net."
            for x in range(weights.shape[0]):
                tot[x]=np.dot(weights[x], input_Tensor)
                output.append(1.0/(1.0+exp(-tot[x]))) 
            input_Tensor=np.delete(input_Tensor,0,0)
            self.__a['W%i' %(layer-1)]=input_Tensor    #Add input tensor to the output dictionary
            input_Tensor=output
            input_Tensor=np.array(input_Tensor)
            output=np.array(output)
            #output=np.insert(output,0,1)        # Append the first row of output to accomodate the bias
            self.__a['W%i' %layer]=output   #Add all the outputs to dictionary
        #print(self.__a)
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
    
    def __dSigma(self,layer):
        return (self.__a['W%i'%(layer)]*(1-self.__a['W%i'%layer]))
    
    
    # Backpropagation pass single target 
    def backward(self, target):
        for x in reversed (range(len(self.__network))):
           x=x+1
           previouslayerout=np.insert(self.__a['W%i'%(x-1)],0,1)[np.newaxis] #Add 1 for bias
           # Compute the partial derivatives for the output layer
           if x==(len(self.__a)-1):             # Length of network is 2, however the length of output is 3(2 outputs+1input tensor) therefore we need 1 less 
               assert target.shape[0]==self.__a['W%i'%x].shape[0], "Please check the number of output neurons and target values"
               delta=np.multiply((self.__a['W%i'%x]-target),self.__dSigma(x))
           else:
               delta =(np.dot(np.delete(self.__network['W%i'%(x+1)],0,0),delta))*self.__dSigma(x)
           # Add new dimension to y and make it an array
           y=np.array(delta)[np.newaxis]
           self.__dE_dTheta['W%i'%x]=np.dot(previouslayerout.T,y)
        #print(self.__dE_dTheta)
                  
   
    def updateParams(self,eta,dtype=float):
        for i in reversed(range(len(self.__network))):
            self.__network['W%i'%(i+1)]=(self.__network['W%i'%(i+1)]-eta*self.__dE_dTheta['W%i'%(i+1)])
        print(self.__network)
        #return(self.__network)
            


