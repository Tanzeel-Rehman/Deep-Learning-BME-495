#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:42:41 2018

@author: tanzeel
"""
from torchvision import transforms
import cv2     #This is to read and preprocess the image fo forward function
import numpy as np
from img2num import img2Num

#Test Script   
num=img2Num()
num.train()
i=cv2.imread('0.png')
i=np.expand_dims(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY),axis=2)
prepro=transforms.Compose([transforms.ToTensor()])
i=prepro(i)
predict=num.forward(i)