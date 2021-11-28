#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:42:41 2018

@author: tanzeel
"""
import torch
from torchvision import transforms
from torch.autograd import Variable
import cv2,os,sys,argparse 
from AlexNet import AlexNet

#Sanity Check
#root='/home/tanzeel/BME 495/Tiny-tiny-imagenet-2'

# Argument Parser Object
parse=argparse.ArgumentParser(description="Testing the Best AlexNet on Tiny ImageNet")
parse.add_argument('--model',type=str)
args=parse.parse_args()


#------------Test Script----------#   
class Test_AlexNet():
    def __init__(self):
        #Initialize the Net
        self.__net=AlexNet()
        
        # Load the Best Stored Model for perdiction
        path_net=os.path.join(args.model,'Best_AlexNet.pth.tar')
        if os.path.isfile(path_net):
            load_net=torch.load(path_net)
            #Call the parameters from Best Model
            self.__net.load_state_dict(load_net['state_dict'])
            self.__numeric_classes=load_net['alpha_classes']
            self.__small_classes=load_net['small_classes']
            epoch=load_net['epoch']
            best_accuracy=load_net['best_accuracy']
            
            print('#-------------Completed Loading the Model-----------#')
            print('The loaded model has Epoch_Best: {}, Best_Accuracy: {}'.format(epoch,best_accuracy))
             
        else:
            print('No saved model found')
            sys.exit(0)
        
    # forward for perdiction
    def forward(self,img):
        self.__net.eval()
        out=self.__net.forward(Variable(img.unsqueeze(0)))
        prediction=out.data.max(1)[1]
        category=self.__small_classes[self.__numeric_classes[prediction[0]]]
        return(category)  
    
    # View one object and its caption
    def cam(self,*idx):
        self.__net.eval()   #Change the model mode to eval
        if not idx:
            idx=0
        else:
            idx=idx[0]
        # Check if the Camera is opened or not
        cap=cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise IOError("Can't open webcam. Please check the webcam index")
        print('\n#-----------Press Esc to Exit------------#')
        
        #Define the font for display
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Define the Preprocessing protocols (tensor,resize,normalize)
        prepro=transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        
        #prepro=transforms.Compose([transforms.ToTensor()])
        cap.set(3,1020)
        cap.set(4,720)
        
        while True:
            ret, frame=cap.read()
            #frame1=cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA)
            frame1=prepro(frame)
            predicted_class=self.forward(frame1) 
            #print(predicted_class)
            cv2.putText(frame,predicted_class,(100,100),font,2,(0,0,255),4,cv2.LINE_AA)
            cv2.imshow('LIVE WEBCAM FEED',frame)
            c=cv2.waitKey(1) 
            if c==27:   # quit the camera stream using the esc, 27 is decimal value for the esc 
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    #i=cv2.imread('val_68.JPEG')
    #prepro=transforms.Compose([transforms.ToPILImage(),transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    #i=prepro(i)     
    mod=Test_AlexNet()
    cat=mod.cam()
    