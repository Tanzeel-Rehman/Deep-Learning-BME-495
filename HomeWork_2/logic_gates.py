# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:23:50 2018

@author: Tanzeel
"""
import numpy as np
from neural_network import NeuralNetwork
# AND Class of Secondary API
class AND():
    def __init__(self):
        self.nn=NeuralNetwork(2,1)
        theta=self.nn.getLayer(0)
        theta[0][0]=-15
        theta[1][0]=10
        theta[2][0]=10
        #print(theta)
    
    def __call__(self,inp1,inp2):
        return self.forward(inp1,inp2)
    
    def forward(self, inp1,inp2):
        inp1=int(inp1==True)
        inp2=int(inp2==True)
        input_Tensor=np.array([inp1,inp2])
        z=self.nn.forward(input_Tensor)
        if z > 0.5:
            return True
        else:
            return False
# OR Class of Secondary API
class OR():
    def __init__(self):
        self.nn=NeuralNetwork(2,1)
        theta1=self.nn.getLayer(0)
        theta1[0][0]=-5
        theta1[1][0]=10
        theta1[2][0]=10
        #print(theta)
    
    def __call__(self,inp1,inp2):
        return self.forward(inp1,inp2)
    
    def forward(self, inp1,inp2):
        inp1=int(inp1==True)
        inp2=int(inp2==True)
        input_Tensor=np.array([inp1,inp2])
        z1=self.nn.forward(input_Tensor)
        if z1 > 0.5:
            return True
        else:
            return False

# NOT Class of Secondary API
class NOT():
    def __init__(self):
        self.nn=NeuralNetwork(1,1)
        theta2=self.nn.getLayer(0)
        theta2[0][0]=5
        theta2[1][0]=-10
            
    def __call__(self,inp1):
        return self.forward(inp1)
    
    def forward(self, inp1):
        inp1=int(inp1==True)
        input_Tensor=np.array([inp1])
        z2=self.nn.forward(input_Tensor)
        if z2 > 0.5:
            return True
        else:
            return False
        
# EXOR Class of Secondary API
class XOR():
    def __init__(self):
        self.nn=NeuralNetwork(2,2,1)
        theta_1=self.nn.getLayer(0)
        theta_1[0][0]=-10
        theta_1[1][0]=20
        theta_1[2][0]=20
        theta_1[0][1]=30
        theta_1[1][1]=-20
        theta_1[2][1]=-20
        #print(theta_1)
        theta_2=self.nn.getLayer(1)
        theta_2[0][0]=-30
        theta_2[1][0]=20
        theta_2[2][0]=20
        #print(theta_2)
    def __call__(self,inp1,inp2):
        return self.forward(inp1,inp2)
    
    def forward(self, inp1,inp2):
        inp1=int(inp1==True)
        inp2=int(inp2==True)
        input_Tensor=np.array([inp1,inp2])
        z1=self.nn.forward(input_Tensor)
        if z1 > 0.5:
            return True
        else:
            return False
