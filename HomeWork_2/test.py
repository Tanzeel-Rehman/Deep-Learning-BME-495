# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:20:19 2018

@author: Tanzeel
"""
# Testing the NeuralNetwork API
import numpy as np
from neural_network import NeuralNetwork
# Net for 1D forward pass
nn=NeuralNetwork(2,2,1)
input_Tensor=np.array([[1],[1]])
# Net for 2D forward pass
nn1=NeuralNetwork(4,4,2)
input_Tensor1=np.array([[1.0,1.0],[1.0,0.0],[1.0,1.0],[0.0,0.0]])
# Forward passes
z=nn.forward(input_Tensor)
z2d=nn1.forward2D(input_Tensor1)
print('This is the output of 1D input tensor:\n', z)
print('This is the output of 2D input tensor:\n', z2d)

# Testing the Logic gates API
#import modules
from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR
# Intialize the gates
And= AND()
Or=OR()
Not=NOT()
Xor=XOR()
# Print the results
print('\nThe results of AND gate (True,False):',And(True,False))
print('The result of OR gate (False,True):', Or(False,True))
print('The result of NOT gate (True):',Not(True))
print('The result of EXOR gate (True,False):', Xor(True,False))