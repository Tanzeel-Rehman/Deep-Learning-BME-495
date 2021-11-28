#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:36:38 2018

@author: tanzeel
"""

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import transforms
from LeNet5 import LeNet
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import time
import cv2

class img2Num():
    def __init__(self):
        root='/home/tanzeel/BME 495/MINIST DATSET'
        # Convert img to tnesor and normalize each band of the image with 0.5 mean and 0.5 stand deviation
        trans=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.__batch=60
        # load the data
        self.__train_loader=torch.utils.data.DataLoader(dset.MNIST(root,train=True,transform=trans,download=False),self.__batch,shuffle=True)
        self.__test_loader=torch.utils.data.DataLoader(dset.MNIST(root,train=False,transform=trans,download=False),self.__batch,shuffle=False)
        self.__classes=('0','1','2','3','4','5','6','7','8','9')     
        #Create the net object of type LeNet5
        torch.manual_seed(1)
        self.__net=LeNet()
        
    def train(self):
        start=time.time()
        self.__net.train()  # Change the mode to train 
        optimizer=optim.SGD(self.__net.parameters(),lr=0.001,momentum=0.9)
        criterion=nn.CrossEntropyLoss()
        correct_test,correct_train,total_test,total_train=0.0,0.0,0.0,0.0
        for epoch in range(1):
            train_loss,test_loss=0.0,0.0
            for i, (img,target) in enumerate(self.__train_loader):
                optimizer.zero_grad()
                img,target=Variable(img),Variable(target)
                out=self.__net.forward(img)
                loss=criterion(out,target)
                predict_train=out.data.max(1)[1]
                total_train+=target.size(0)
                correct_train+=predict_train.eq(target.data.view_as(predict_train)).sum()
                loss.backward()
                optimizer.step()
                if (i+1)%self.__batch==0 or (i+1)==len(self.__train_loader):
                    print('=> epoch: {},batch index: {},Accuracy_train: {:.6f},train loss:{:.6f}'.format(epoch+1,(i+1),100*correct_train/total_train,loss.data[0]))
                    train_loss+=loss.data[0]
            train_loss=train_loss*self.__batch/(len(self.__train_loader))
            end=time.time()
            #Test the model for every epoch
            self.__net.eval()   # Change the mode to val. This is imp, if not the training net will be used, which causes the random masking for dropout
            for (img,target) in (self.__test_loader):
                img,target=Variable(img),Variable(target)
                output=self.__net.forward(img)
                test_loss+=criterion(output,target).sum()
                pred=output.data.max(1)[1]
                total_test+=target.size(0)
                correct_test += pred.eq(target.data.view_as(pred)).sum()
            print('==> epoch:{}, Accuracy:{:.6f},test_loss: {:.6f}'.format(epoch+1,100*correct_test/total_test,(test_loss.data[0]/len(self.__test_loader))))
            # Plot the loss against the epoch
            plt.plot(epoch,train_loss,'bo',epoch,test_loss.data[0]/len(self.__test_loader),'r+')   #Plot the Mean error against the epochs
            plt.title('Trainning and Test Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train','Test'],loc='upper right')
        plt.show()
        print('Trainning Time: {}'.format(end-start))
    # forward for perdiction
    def forward(self,img):
        start=time.time()
        self.__net.eval()
        y=self.__net.forward(Variable(img.unsqueeze(0)))
        prediction=y.data.max(1)[1]
        category=self.__classes[prediction[0]]
        end=time.time()
        print('Inference Time:{}'.format(end-start))
        return(int(category))       


