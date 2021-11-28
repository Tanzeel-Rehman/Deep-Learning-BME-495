
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:08:28 2018

@author: Tanzeel
"""
import numpy as np
from math import ceil

'''Base class for performing convolution operation on images'''
class Conv2D:
    def __init__(self,in_channel=0,out_channel=0,kernel_size=0,stride=0,mode='known'):
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.mode=mode
        
    def forward(self,input_image):
        
        if self.out_channel==1 and self.mode=='known':
            kernel=np.array([(-1,-1,-1),(0,0,0),(1,1,1)]) 
            kernel=np.flipud(kernel)
            kernel=kernel[np.newaxis,np.newaxis,...]
            kernel=np.repeat(kernel,self.in_channel,axis=1)
            
                        
        elif self.out_channel==3 and self.mode=='known':
            # 1st kernel
            kernel1=np.array([(-1,-1,-1),(0,0,0),(1,1,1)])
            # Flip the kernel
            kernel1=np.flipud(kernel1)
            kernel1=kernel1[np.newaxis,...]
            kernel1=np.repeat(kernel1,self.in_channel,axis=0)
            #2nd Kernel
            kernel2=np.array([(-1,0,1),(-1,0,1),(-1,0,1)])
            #Flip the kernel
            kernel2=np.fliplr(kernel2)
            kernel2=kernel2[np.newaxis,...]
            kernel2=np.repeat(kernel2,self.in_channel,axis=0)
            #3rd kernel
            kernel3=np.array([(1,1,1),(1,1,1),(1,1,1)])
            kernel3=kernel3[np.newaxis,...]
            kernel3=np.repeat(kernel3,self.in_channel,axis=0)
            
            # Combine the kernels
            kernel=np.zeros_like(kernel1,dtype=np.float)
            kernel=kernel[np.newaxis,...]
            kernel=np.repeat(kernel,self.out_channel,axis=0)
            kernel[0,...]=kernel1
            kernel[1,...]=kernel2
            kernel[2,...]=kernel3
        elif self.out_channel==2 and self.mode=='known':
            # 1st kernel
            kernel1=np.array([(-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(0,0,0,0,0),(1,1,1,1,1),(1,1,1,1,1)])
            #Flip the kernel
            kernel1=np.flipud(kernel1)
            kernel1=kernel1[np.newaxis,...]
            kernel1=np.repeat(kernel1,self.in_channel,axis=0)
            #2nd Kernel
            kernel2=np.array([(-1,-1,0,1,1),(-1,-1,0,1,1),(-1,-1,0,1,1),(-1,-1,0,1,1),(-1,-1,0,1,1)])
            #Flip the kernel
            kernel2=np.fliplr(kernel2)
            kernel2=kernel2[np.newaxis,...]
            kernel2=np.repeat(kernel2,self.in_channel,axis=0)
            # Combine the kernels
            kernel=np.zeros_like(kernel1,dtype=np.float)
            kernel=kernel[np.newaxis,...]
            kernel=np.repeat(kernel,self.out_channel,axis=0)
            kernel[0,...]=kernel1
            kernel[1,...]=kernel2
            
        elif self.mode=='rand':
            kernel=np.random.randint(-self.kernel_size,self.kernel_size,size=(self.out_channel,self.in_channel,self.kernel_size,self.kernel_size))
        
        
        assert len(kernel.shape)==4, "The size of kernel should be [# of output_channels, #input_channels,height, width]" 
        assert len(input_image.shape)==3, "The size of input image should be [# of in_channels,height,width]"
        assert kernel.shape[1]==self.in_channel, "The image and kernel should have the same number of input_channels"
       
        input_w,input_h=input_image.shape[2], input_image.shape[1]
        kernel_w, kernel_h=self.kernel_size, self.kernel_size
        
        k=self.kernel_size
        counter=0
        output_height=int(ceil(float(input_h)/float(self.stride)))
        # if self.padding=='same':   (The padding mechanism used here is adopted from the Tensor Flow)
        output_height=int(ceil(float(input_h)/float(self.stride)))
        output_width=int(ceil(float(input_w)/float(self.stride)))
       # Calculating the number of zeros required for padding
        pad_height=max((output_height-1)*self.stride +kernel_h -input_h,0)
        pad_width=max((output_width-1)*self.stride +kernel_w -input_w,0)
        pad_top=pad_height//2
        pad_bottom=pad_height-pad_top
        pad_left=pad_width//2
        pad_right=pad_width-pad_left
        output=np.zeros(( self.out_channel,output_height,output_width))
            
        #Add zeros padding to the input image
        image_padded=np.zeros((input_image.shape[0],input_image.shape[1]+pad_height,input_image.shape[2]+pad_width))
        image_padded[:,pad_top:-pad_bottom,pad_left:-pad_right]=input_image
        #Loop over every pixel
        for ch in range(self.out_channel):
                for x in range(output_width):
                    for y in range (output_height):
                        output[ch,y,x]=(kernel[ch,...]*image_padded[:,y*self.stride:y*self.stride+kernel_h,x*self.stride:x*self.stride+kernel_w]).sum()
                        counter=counter+((k*k*self.in_channel)+(k*k*self.in_channel)-1)
        ''' elif self.padding == 'valid':
            output_height = int(ceil(float(input_h - kernel_h + 1) / float(self.stride)))
            output_width = int(ceil(float(input_w - kernel_w + 1) / float(self.stride)))
            output = np.zeros((output_height, output_width, self.out_channel))  # convolution output
        
            for ch in range(self.out_channel):
                for x in range(output_width):  # Loop over every pixel of the output
                    for y in range(output_height):
                    # element-wise multiplication of the kernel and the image
                        output[ch,y,x]=(kernel[ch,...]*image_padded[:,y*self.stride:y*self.stride+kernel_h,x*self.stride:x*self.stride+kernel_w]).sum()
                        counter=counter+((k*k)+(k*k-1))*(self.in_channel)'''
        return (counter,output)
