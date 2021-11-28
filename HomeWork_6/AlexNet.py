#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  18 16:35:58 2018

@author: Tanzeel
"""


import torch.nn as nn
from torchvision import models




class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        """
        Define the Model
        """
        '''
        self.features=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )'''
        # Download  the pretrained model
        self.pretrained_net=models.alexnet(pretrained=True)
        '''self.classifier = nn.Sequential(nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 200),
        )'''
        self.classifier = nn.Sequential(
            self.pretrained_net.classifier[0],
            self.pretrained_net.classifier[1],
            self.pretrained_net.classifier[2],
            self.pretrained_net.classifier[3],
            self.pretrained_net.classifier[4],
            self.pretrained_net.classifier[5],
            nn.Linear(4096,200),    # Change the last layer only
        )
    def forward(self,img):
        """
        Execute the forward pass
        """
        #img=self.features(img)
        img=self.pretrained_net.features(img)
        img=img.view(img.size(0),-1)      # reshape the input image to 1D tensor
        img=self.classifier(img)
        return img