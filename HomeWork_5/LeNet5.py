#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  18 16:35:58 2018

@author: Tanzeel
"""


import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        #Setup the architecture of the net
        #self.C1=nn.Conv2d(1,6,5,padding=2)
        #self.C2=nn.Conv2d(6,16,5)
        #self.C3=nn.Conv2d(16,120,5)
        #self.fc2=nn.Linear(120,84)
        #self.fc3=nn.Linear(84,10)
        self.Conv_LeNet=nn.Sequential(nn.Conv2d(1,6,5,padding=0),nn.MaxPool2d(2),
                nn.Tanh(),nn.Conv2d(6,16,5),nn.MaxPool2d(2),nn.Conv2d(16,120,5))
        self.fc_LeNet=nn.Sequential(nn.Linear(120,84),nn.Tanh(),nn.Linear(84,10),nn.Tanh())
        
    def forward(self,img):
        #img=self.C1(img)
        #img=F.tanh(F.max_pool2d(img,2))
        #img=self.C2(img)
        #img=F.tanh(F.max_pool2d(img,2))
        #img=self.C3(img)
        #img=img.view(img.size(0),-1) # reshape the input image to 1D tensor
        #img=F.tanh(self.fc2(img))
        #img=F.tanh(self.fc3(img))
        img=self.Conv_LeNet(img)
        img=img.view(img.size(0),-1)      # reshape the input image to 1D tensor
        img=self.fc_LeNet(img)
        return img