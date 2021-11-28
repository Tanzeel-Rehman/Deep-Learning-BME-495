#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  18 16:35:58 2018

@author: Tanzeel
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time

class NnImg2Num(nn.Module):
    def __init__(self):
        #Download thr MINIST dataset
        root='/home/tanzeel/BME 495/MINIST DATSET'
        # Convert img to tnesor and normalize each band of the image with 0.5 mean and 0.5 stand deviation
        trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train_set=dset.MNIST(root=root,train=True,transform=trans,download=False) #Don't download for online trainning
        test_set=dset.MNIST(root=root,train=False,transform=trans,download=False)
        self.__batch=60
        # load the data
        self.train_loader=torch.utils.data.DataLoader(dataset=train_set,batch_size=self.__batch,shuffle=True)
        self.test_loader=torch.utils.data.DataLoader(dataset=test_set,batch_size=self.__batch,shuffle=True)
        self.classes=('0','1','2','3','4','5','6','7','8','9')
        super(NnImg2Num,self).__init__()
        #Setup the architecture of the net
        self.fc1=nn.Linear(28*28,100,bias=True)
        self.fc2=nn.Linear(100,50,bias=True)
        self.fc3=nn.Linear(50,30,bias=True)
        self.fc4=nn.Linear(30,10,bias=True)
    def forward(self,img):
        img=img.view(-1,28*28) # reshape the input image to 1D tensor
        img=F.sigmoid(self.fc1(img))
        img=F.sigmoid(self.fc2(img))
        img=F.sigmoid(self.fc3(img))
        img=F.sigmoid(self.fc4(img))
        _, pred=torch.max(img.data,1)     #Perdict the output
        return img
    def train(self):
        self.__model=NnImg2Num()
        optimizer=optim.SGD(self.__model.parameters(),lr=0.001,momentum=0.9)
        criterion=nn.MSELoss(size_average=False)
        y_onehot=torch.FloatTensor(self.__batch,10) # 10=number of digits
        for epoch in range(100):
            error=[]
            for batch_idx, (img,target) in enumerate(self.train_loader):
                start=time.time()
                optimizer.zero_grad()
                img=Variable(img)
                target=target.type(torch.LongTensor).view(-1,1)
                y_onehot.zero_()
                y_onehot.scatter_(1,target,1)
                out=self.__model.forward(img)
                target=Variable(y_onehot)
                loss=criterion(out,target)
                loss_y=loss.data.numpy()
                loss.backward()
                optimizer.step()
                end=time.time()
                if (batch_idx+1)%self.__batch==0 or (batch_idx+1)==len(self.train_loader):
                    print('=> epoch: {},batch index: {},train loss:{:.6f}, time:{}'.format(epoch+1,(batch_idx+1),loss.data[0],(end-start)))
                    error.append(np.average(loss_y))      
                    #plt.plot(epoch,error[0],'ro')   #Plot the error against the epochs
                    plt.plot(epoch,(end-start),'ro')
            plt.show()
    '''
    # Check the acuracy on test dataset
    def test(self):
        correct,total=0.0,0.0
        for (img,target) in self.test_loader:
            img=Variable(img,volatile=True)
            output=self.__model(img)
            _, pred=torch.max(output.data,1)
            total+=target.size(0)
            correct += (pred==target).sum()
        print('Accuracy of the network on test data is:%d %%' %(100*correct/total))
    #Check the accuracy of individual class
    def individual(self):
        class_correct=list(0. for i in range(10))
        class_total=list(0. for i in range(10))
        for (img,target) in self.test_loader:
            img=Variable(img,volatile=True)
            output=self.__model(img)
            _, pred=torch.max(output.data,1)
            c=(pred==target).squeeze()
            for i in range(4):
                label=target[i]
                class_correct[label]+=c[i]
                class_total[label] +=1
        for i in range(10):
            print('Accuracy of%5s is: %2d%%'%(self.classes[i],100*class_correct[i]/class_total[i]))
'''
# Run the model
#model1=NnImg2Num()
#model1.train()
#model1.test()
#model1.individual()