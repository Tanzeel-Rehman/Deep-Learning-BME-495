#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:36:38 2018

@author: Tanzeel
"""
import torch,cv2
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from LeNetObj import LeNet
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import time

class img2obj():
    def __init__(self):
        root='/home/tanzeel/BME 495/CIFAR10'
        # Convert img to tnesor and normalize each band of the image with 0.5 mean and 0.5 stand deviation
        trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.__batch=64
        # load the data
        self.__train_loader=torch.utils.data.DataLoader(dset.CIFAR10(root,train=True,transform=trans,download=False),self.__batch,shuffle=True)
        self.__test_loader=torch.utils.data.DataLoader(dset.CIFAR10(root,train=False,transform=trans,download=False),self.__batch,shuffle=False)
        self.__classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')     
        #Create the net object of type LeNet5
        #torch.manual_seed(1)
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
            plt.plot(epoch,train_loss,'bo',epoch,(test_loss.data[0]/len(self.__test_loader)),'r+')   #Plot the Mean error against the epochs
            plt.title('Trainning and Test Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train','Test'],loc='upper right')
            #plt.subplot(212)
            #plt.plot(epoch,(test_loss.data[0]/len(self.__test_loader)),'r+')   #Plot the Mean error against the epochs
            #plt.ylabel('Test Loss')
            #plt.xlabel('Epoch')
            #plt.legend(['Test'],loc='upper right')
        plt.show()
        print(end-start)
    # forward for perdiction
    def forward(self,img):
        self.__net.eval()   #Change the model mode to eval
        y=self.__net.forward(Variable(img.unsqueeze(0))) # Make the tensor as variable and execute the forward pass
        prediction=y.data.max(1)[1]
        #prediction=np.asscalar(prediction.numpy())      
        return(self.__classes[prediction[0]])       
    
    # View one object and its caption
    def view(self,img):
        self.__net.eval()   #Change the model mode to eval
        y=self.__net.forward(Variable(img.unsqueeze(0))) # Make 4D tensor(batch x CxHxW) and convert to variable
        prediction=y.data.max(1)[1]
        category=self.__classes[prediction[0]]
        plt.figure(2)
        #Convert the tensor to img and change color from BGR to RGB
        img=cv2.cvtColor(img.numpy().transpose(1,2,0),cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.text(32,2,category,color='r',fontsize=20)
        plt.show()
    
    # View one object and its caption
    def cam(self,*idx):
        self.__net.eval()   #Change the model mode to eval
        if not idx:
            idx=0
        else:
            idx=idx[0]
        cap=cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise IOError("Can't open webcam")
        while True:
            ret, frame=cap.read()
            frame1=cv2.resize(frame,(32,32),interpolation=cv2.INTER_AREA)
            prepro=transforms.Compose([transforms.ToTensor()])
            frame1=prepro(frame1)
            y=self.__net.forward(Variable(frame1.unsqueeze(0))) # Make 4D tensor(batch x CxHxW) and convert to variable
            predict=y.data.max(1)[1]
            cate=""    # Make string null to remove any previous predictions 
            cate=self.__classes[predict[0]]
            #print(cate)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,cate,(100,100),font,4,(0,0,255),4)
            cv2.imshow('Input',frame)
            c=cv2.waitKey(1)
            if c==27:   # quit the camera stream using the esc, 27 is decimal value for the esc 
               break
        cap.release()
        cv2.destroyAllWindows()
    
    '''def individual(self):
        self.__net.eval()
        class_correct=list(0. for i in range(10))
        class_total=list(0. for i in range(10))
        for (img,target) in self.__test_loader:
            img=Variable(img,volatile=True)
            output=self.__net(img)
            _, pred=torch.max(output.data,1)
            c=(pred==target).squeeze()
            for i in range(4):
                label=target[i]
                class_correct[label]+=c[i]
                class_total[label] +=1
        for i in range(10):
            print('Accuracy of %5s is: %2d%%'%(self.__classes[i],100*class_correct[i]/class_total[i]))
        '''
