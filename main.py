# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:45:46 2018

@author: Tanzeel
"""

'''Import necessary modules'''
from scipy.misc import imread
import matplotlib.pyplot as plt
# Import the Conv2D class 
from conv import Conv2D


def main():
    image=imread('Image_1.png',mode='RGB') 
    # Image is parsed as (# of in_channels, H, W)  
    image=image.transpose((2,0,1))
    # Initialize the Conv2D and call the forward function
    conv2d=Conv2D(in_channel=3,out_channel=2,kernel_size=5,stride=1,mode='known')
    operations,output=conv2d.forward(image)
    print('Number of operations:',operations)              # print the number of operations 
    print ('Output image tensor size: ', output.shape)     # Print the output image dimensions 
    print ('main.py is currently running kernels for task 3, please change the parameters for other tasks')
    # plot these images
    plt.imshow(output[0,...],cmap=plt.cm.gray),plt.show()
    plt.imshow(output[1,...],cmap=plt.cm.gray),plt.show()
    #plt.imshow(output[2,...],cmap=plt.cm.gray),plt.show()
    
if __name__ == "__main__":
    main()