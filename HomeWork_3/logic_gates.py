# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:23:50 2018

@author: Tanzeel
"""
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
# AND Class of Secondary API
class AND():
    def __init__(self):
        self.nn=NeuralNetwork(2,1)
    
    def forward(self, inp1,inp2):
        inp1=int(inp1==True)
        inp2=int(inp2==True)
        input_Tensor=np.array([inp1,inp2])
        z=self.nn.forward(input_Tensor)
        if z > 0.5:
            return True
        else:
            return False
    
    def train(self):
        epochs=1000
        for epoch in range(epochs):
            error=0.0
            sum_error=[]
            x1 = [False, False, True, True]
            x2 = [False, True, False, True]
            for i in range(4):
                if x1[i] and x2[i]:
                    target=True
                else:
                    target=False
                inp1=int(x1[i]==True)
                inp2=int(x2[i]==True)
                input_tensor=np.array([inp1,inp2])
                z=self.nn.forward(input_tensor)
                target=int(target==True) # Convert the target to int for error calculation
                error= (0.5*(target-z)**(2)).sum() # Calculate the total error
                sum_error.append(error)
                self.nn.backward(target)
                self.nn.updateParams(0.5)
                print('>epoch=%d, error=%0.7f,1st_input=%i,2nd_input=%i,output=%0.7f,target=%i' %(epoch,error,inp1,inp2,z,target))
                plt.plot(epoch,sum_error[i],'ro')   #Plot the error against the epochs
        plt.show()

    
# OR Class of Secondary API
class OR():
    def __init__(self):
        self.nn=NeuralNetwork(2,1)
    
    def forward(self, inp1,inp2):
        inp1=int(inp1==True)
        inp2=int(inp2==True)
        input_Tensor=np.array([inp1,inp2])
        z=self.nn.forward(input_Tensor)
        if z > 0.5:
            return True
        else:
            return False
    
    def train(self):
        epochs=1000
        for epoch in range(epochs):
            error=0.0
            sum_error=[]
            x1 = [False, False, True, True]
            x2 = [False, True, False, True]
            for i in range(4):
                if x1[i] or x2[i]:
                    target=True
                else:
                    target=False
                inp1=int(x1[i]==True)
                inp2=int(x2[i]==True)
                input_tensor=np.array([inp1,inp2])
                z=self.nn.forward(input_tensor)
                target=int(target==True) # Convert the target to int for error calculation
                error= (0.5*(target-z)**(2)).sum() # Calculate the total error
                sum_error.append(error)
                self.nn.backward(target)
                self.nn.updateParams(0.5)
                print('>epoch=%d, error=%0.7f,1st_input=%i,2nd_input=%i,output=%0.7f,target=%i' %(epoch,error,inp1,inp2,z,target))
                plt.plot(epoch,sum_error[i],'ro')
        plt.show()
        
# NOT Class of Secondary API
class NOT():
    def __init__(self):
        self.nn=NeuralNetwork(1,1)
    
    def forward(self, inp1,inp2):
        inp1=int(inp1==True)
        inp2=int(inp2==True)
        input_Tensor=np.array([inp1,inp2])
        z=self.nn.forward(input_Tensor)
        if z > 0.5:
            return True
        else:
            return False
    
    def train(self):
        epochs=1000
        for epoch in range(epochs):
            error=0.0
            sum_error=[]
            # This is random data generation protocol for NOT gate
            #np.random.seed(1)
            x1 = [False, True]
            #y = [False, False, False, True]
            #input_tensor=np.random.choice([True,False],(2))
            for i in range(2):
                target=not(x1[i])
                target=int(target==True) # Convert the target to int for error calculation
                inp1=int(x1[i]==True)
                input_tensor=np.array([inp1])
                z=self.nn.forward(input_tensor)
                error= (0.5*(target-z)**(2)).sum() # Calculate the total error
                sum_error.append(error)
                self.nn.backward(target)
                self.nn.updateParams(0.5)
                print('>epoch=%d, error=%0.7f,1st_input=%i,output=%0.7f,target=%i' %(epoch,error,inp1,z,target))
                plt.plot(epoch,sum_error[i],'ro')
        plt.show()


# AND Class of Secondary API
class XOR():
    def __init__(self):
        self.nn=NeuralNetwork(2,2,1)
    
    def forward(self, inp1,inp2):
        inp1=int(inp1==True)
        inp2=int(inp2==True)
        input_Tensor=np.array([inp1,inp2])
        z=self.nn.forward(input_Tensor)
        if z > 0.5:
            return True
        else:
            return False
    
    def train(self):
        epochs=40000
        for epoch in range(epochs):
            error=0.0
            sum_error=[]
            
            # This is random data generation protocol for OR gate
            #np.random.seed(1)
            x1 = [False, False, True, True]
            x2 = [True, False, False, True]
            x1=np.random.choice(x1,1)
            x2=np.random.choice(x2,1)
            target=(not(x1 and x2))and((x1 or x2))
            inp1=int(x1==True)
            inp2=int(x2==True)
            input_tensor=np.array([inp1,inp2])
            z=self.nn.forward(input_tensor)
            target=int(target==True) # Convert the target to int for error calculation
            error=((0.5*(target-z)**(2)).sum())
            sum_error.append(error) # Calculate the total error
            self.nn.backward(target)
            self.nn.updateParams(0.001)
            print('>epoch=%d, error=%0.7f, 1st_input=%i,2nd_input=%i,output=%0.7f,target=%i' %(epoch,error,inp1,inp2,z,target))
            plt.plot(epoch,sum_error,'ro')
        plt.show()
           