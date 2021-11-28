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
        for i in range (1, len(args)):
            #np.random.seed(1)
            layers = np.random.normal(0.00,(1/np.sqrt(args[i-1])),(args[i-1]+1,args[i]))
            #layers=layers.transpose()
            self.__network['W%i'%i]=layers
        
    def getLayer(self,layer, dtype=int):
        layer=layer+1
        return (self.__network['W%i'%layer])
    
    
    # More efficient implwmwntation of 1D forward pass
    def forward(self,input_Tensor):
        
        for layer in self.__network:
            weights=self.__network[layer]
            weights=weights.transpose()
            output=[]
            tot=np.zeros(weights.shape[0])
            input_Tensor=np.insert(input_Tensor,0,1)
            assert weights.shape[1]==input_Tensor.shape[0], "Incorrect input Tensor. Tensor height should be similar to the input dimensions used while intializing the Net."
            for x in range(weights.shape[0]):
                tot[x]=np.dot(weights[x], input_Tensor)
                output.append(1.0/(1.0+exp(-tot[x]))) 
            input_Tensor=output
            input_Tensor=np.asarray(input_Tensor)
            output=np.asarray(output)
        return (output)
    
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
        return (output)




