#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 18 23:36:38 2018

@author: tanzeel
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models
from AlexNet import AlexNet
from dataloader import datasetup
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import time,argparse,os

#Sanity Check
#root='/home/tanzeel/BME495/Tiny-tiny-imagenet-2'

# Argument Parser Object
parse=argparse.ArgumentParser(description="AlexNet transfer learning")
parse.add_argument('--data',type=str)
parse.add_argument('--save',type=str)
args=parse.parse_args()

class Train_AlexNet():
    def __init__(self):
        #----------Data Prepration and Loading----------#
        Datasetup=datasetup()
        
        # Check if the GPU is available
        self.use_gpu=torch.cuda.is_available()
        # Batch size for training and validation
        self.__train_batch=100
        self.__val_batch=10
        
        #Define root folders containing the images for trainning and validation
        train_root=os.path.join(args.data,'train')
        val_root=os.path.join(args.data,'val/images')
        
        # Generate he folders for validation images
        Datasetup.val_classes(args.data)
        
        # Convert img to tnesor and normalize each band of the image with mean and std. deviation defined by pytorch torchvision. models help
        trans_train=transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        
        trans_Val=transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        
        # Create the dataset for trainning and validation by applying the tansforms on data
        train_data=dset.ImageFolder(train_root,transform=trans_train)
        val_data=dset.ImageFolder(val_root,transform=trans_Val)
                
        # load the data
        self.__train_loader=torch.utils.data.DataLoader(train_data,self.__train_batch,shuffle=True,num_workers=4)
        self.__val_loader=torch.utils.data.DataLoader(val_data,self.__val_batch,shuffle=False,num_workers=4)
        
        # Extract the classes as dict of string labels
        self.aplpha_classes=train_data.classes
        self.num_classes=len(self.aplpha_classes)
        self.__small_classes=Datasetup.classes(args.data)     
        
        #----------Download the pretrained AlexNet and Define our own AlexNet (see:e-lab>pytorchtollbox)----------#
        print('\n#-----------Downloading the pretrianed AlexNet-------------#')
        #pretrained_net=models.alexnet(pretrained=True)
        print('#------------Download Completed------------#')
        
        self.__net=AlexNet()
        if self.use_gpu:
            #pretrained_net=pretrained_net.cuda()
            self.__net=self.__net.cuda()
                 
        '''
        #----------Check the few weights of our model and pretrained model-------------#
        print(pretrained_net.classifier[1].weight)
        print(self.__net.classifier[1].weight)
        # These weights are different. Let's copy the weights
        
        #----------Copy the weights from pretrained net to the our own defined AlexNet (help: e-lab>pytorch)----------#
        print('#-----------Copying the Weights from Pretrained AlexNet-------------#')
        for i, j in zip(self.__net.modules(), pretrained_net.modules()):
            if not list(i.children()):
                if len(i.state_dict()) > 0:  # relu and dropout do not have anything in their state_dict
                    if i.weight.size()==j.weight.size():
                        j.weight.data = i.weight.data
                        j.bias.data = i.bias.data
        '''
        
        #First Freeze the weights of entire model (help: pytorch transfer learning tutorial)
        for param in self.__net.parameters():
            param.requires_grad=False
        #Enable the gradient calculation for the last layer
        for param in self.__net.classifier[6].parameters():
            param.requires_grad=True
        print('#----------Weights Copied and Saved----------#')
        # Define the lists for storing the accuracies and loss during train and Val 
        self.Train_loss,self.Val_loss=[],[]
        self.Train_Accu,self.Val_Accu=[],[]
        #Define the hyper-parameters for the trainning
        self.__criterion=nn.CrossEntropyLoss()
        # Define the optimizer, but make sure that you specified only the last layer for optimization rather than whole net
        self.__optimizer=optim.Adam(self.__net.classifier[6].parameters(),lr=0.001)
        # Define intial best accuracy for benchamrking the next accuracies to store the models
        self.__best_accuracy=0.0
            
    #----------Functions trainning the Entire net For All epochs----------#
    def train_EntireNet(self):
        
       # Train the Net for 1 Epoch
        def __train_OneEpoch(epochs):
            start=time.time()
            self.__net.train()  # Change the mode to train 
            correct_train,total_train=0.0,0.0
            train_loss=0.0
            for batch_idx, (img,target) in enumerate(self.__train_loader):
               
                if self.use_gpu:
                    img=Variable(img.cuda())
                    target=Variable(target.cuda(),requires_grad=False)
                else:
                    img,target=Variable(img),Variable(target,requires_grad=False)
                
                # Make the Gradient Update zero
                self.__optimizer.zero_grad()
                
                # Do the forward pass through the model
                out=self.__net.forward(img)
                # Calculate the batch loss and accumulate it for the average loss
                loss=self.__criterion(out,target)
                train_loss+=loss.data[0]
                #Backpropagate the error and update the parameters
                loss.backward()
                self.__optimizer.step()
                
                # Predict the class through the model output
                _,predict_train=torch.max(out.data,1)
                total_train+=target.size(0) 
                correct_train+=torch.sum(predict_train==target.data)
                if (batch_idx+1)%self.__train_batch==0 or (batch_idx+1)==len(self.__train_loader):
                    print('=> epoch: {}, batch index: {}, Batch_Accu_train: {:.6f}, Batch_train_loss:{:.6f}'.format(epochs+1,(batch_idx+1),100*(correct_train/total_train),loss.data[0]))
                        
            #------------The Average Accuracy and Avergae loss will be used for ploting-------------#
            
            #Average out the accumulated error for 1 Epoch         
            train_loss=train_loss*self.__train_batch/(len(self.__train_loader.dataset))
            # Average Accuracy for 1 Epoch
            Average_train_accuracy=100.0*(correct_train/len(self.__train_loader.dataset))
            print('<<=>>epoch:{}, Avg_train_loss: {:.5f}, Avg_train_Accu:{:.5f} '.format(epochs+1, train_loss,Average_train_accuracy))
            end=time.time()
            elapsed=end-start
            return (train_loss,Average_train_accuracy,elapsed)
        
        #----------Evaluate the Trained Net for 1 Epoch---------------#
        def __val_OneEpoch(epochs):
            #Test the model for every epoch
            self.__net.eval()   # Change the mode to val. This is imp, if not the training net will be used, which causes the random masking for dropout
            correct_val,total_val=0.0,0.0
            Average_val_loss=0.0
            for (img,target) in (self.__val_loader):
                if self.use_gpu:
                    img=Variable(img.cuda())
                    target=Variable(target.cuda(),requires_grad=False)
                else:
                    img,target=Variable(img),Variable(target,requires_grad=False)
                
                #Do forward pass through the model
                out=self.__net.forward(img)
                # Calculate the batch loss and accumulate it for the average loss
                loss=self.__criterion(out,target)
                Average_val_loss+=loss.data[0]
                
                # Predict the class through the model output
                _,predict_val=torch.max(out.data,1)
                total_val+=target.size(0) 
                correct_val+=torch.sum(predict_val==target.data)
            print('==> epoch:{}, Batch_Accu_Val:{:.5f}, Batch_Val_loss: {:.5f}'.format(epochs+1,100*correct_val/total_val,loss.data[0]))
            
            #Average out the accumulated error for 1 Epoch         
            Average_val_loss=Average_val_loss*self.__val_batch/(len(self.__val_loader.dataset))
            # Average Accuracy for 1 Epoch
            Average_val_accuracy=100.0*(correct_val/len(self.__val_loader.dataset))
            print('<<=>>epoch:{}, Avg_Val_loss: {:.5f}, Avg_Val_Accu:{:.5f}'.format(epochs+1, Average_val_loss,Average_val_accuracy))
            return (Average_val_loss,Average_val_accuracy)
        
        #--------------Save The Model (help:FloydHub>Save and Resume Your Experiments)-------------#
        def save_checkpoint(state,is_best,filename=os.path.join(args.save,'Best_AlexNet.pth.tar')):
            if is_best:
                print('#----------Saving New Best Model------------#')
                torch.save(state,filename)
            else:
                print('#--------------Validation accuracy did not improve--------------#')
            
        
        # Call the trainning Validation and Save Functions to iterate for certian number of epochs
        print('#----------Starting the trainning process----------#')
        # Define the number of epochs for training
        train_epochs=20
        num_epoch=range(1, train_epochs+1)
        
        for epoch in range(train_epochs):
            T_loss,T_accu,T_Time=__train_OneEpoch(epoch)   
            V_loss,V_accu=__val_OneEpoch(epoch)
            # Append the statistics to th lists
            self.Train_loss.append(T_loss)
            self.Train_Accu.append(T_accu)
            self.Val_loss.append(V_loss)
            self.Val_Accu.append(V_accu)
            # Save the Best Model on the basis of the Va;idation accuracy.Alternatively, Val_loss can also be used
            is_best=V_accu >self.__best_accuracy
            self.__best_accuracy=max(V_accu,self.__best_accuracy)
            save_checkpoint({'epoch':epoch+1,
                             'state_dict':self.__net.state_dict(),
                             'best_accuracy':self.__best_accuracy,
                             'optimizer':self.__optimizer.state_dict(),
                             'train_loss':T_loss,
                             'val_loss':V_loss,
                             'Train_Accuracy':T_accu,
                             'Val_Accuracy':V_accu,
                             'alpha_classes':self.aplpha_classes,
                             'small_classes':self.__small_classes},is_best)
            print('Time for One Epoch:{:.5f}seconds'.format(T_Time))
            print('######################################################################\n')
    
        #------------Start Plotting the Loss and Accuracy------------#
        #Plot the loss Vs Epoch
        plt.figure(1)
        plt.plot(num_epoch,self.Train_loss,'bo',num_epoch,self.Val_loss,'r+')   #Plot the Mean error against the epochs
        plt.title('Trainning and Test Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train','Test'],loc='upper right')
        plt.show()
        #Plot the accuracy
        plt.figure(2)
        plt.plot(num_epoch,self.Train_Accu,'bo',num_epoch,self.Val_Accu,'r+')   #Plot the Mean error against the epochs
        plt.title('Trainning and Test Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train','Test'],loc='lower right')
        plt.show()


if __name__=='__main__':
    net=Train_AlexNet()
    net.train_EntireNet()