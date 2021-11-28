#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:23:08 2018

@author: tanzeel
"""
from torchvision import transforms
import cv2
from img2obj import img2obj

#Test Script   
num=img2obj()
num.train()
i=cv2.imread('1031.png')
prepro=transforms.Compose([transforms.ToTensor()])
i=prepro(i)
predict=num.forward(i)
num.view(i)
num.cam()