# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:44:43 2018

@author: Tanzeel
"""
# imort necessary modules
import time
from scipy.misc import imread,imsave
import matplotlib.pyplot as plt
# Import the class 
from conv import Conv2D

image=imread('Image_2.png',mode='RGB')   
# Image is parsed as (# of in_channels, H, W)  
image=image.transpose((2,0,1))
# Start measuring the time for T1
start_T1=time.time()
#This is the intialization for the first Task
conv2d_T1=Conv2D(in_channel=3,out_channel=1,kernel_size=3,stride=1,mode='known')
operations_T1,output_T1=conv2d_T1.forward(image)
end_T1=time.time()
print('Number of operations for Task 1:',operations_T1)              # print the number of operations 
print ('Output image tensor size for Task 1: ', output_T1.shape)     # Print the output image dimensions 
print('Time (s) to execute the forward pass for Task 1:',end_T1-start_T1)   # Print the processing time
# Indicate the current state of code
print('test.py is still running, please close the picture to continue\n')
# Start subplotting the gray output images. Number of channels in output image vary. 
#This will throw an error if the output image channels are less than 1
plt.imshow(output_T1[0,...],cmap=plt.cm.gray), plt.show()

# Start measuring the time for T2
start_T2=time.time()
#This is the intialization for the T2
conv2d_T2=Conv2D(in_channel=3,out_channel=2,kernel_size=5,stride=1,mode='known')
operations_T2,output_T2=conv2d_T2.forward(image)
end_T2=time.time()
print('Number of operations for Task 2:',operations_T2)              # print the number of operations 
print ('Output image tensor size for Task 2: ', output_T2.shape)     # Print the output image dimensions 
print('Time (s) to execute the forward pass for Task 2:',end_T2-start_T2)   # Print the processing time
# Indicate the current state of code
print('test.py is still running, please close all the pictures to continue\n')
# Start subplotting the gray output images. Number of channels in output image vary. 
#This will throw an error if the output image channels are less than 2
plt.imshow(output_T2[0,...],cmap=plt.cm.gray),plt.show()
plt.imshow(output_T2[1,...],cmap=plt.cm.gray),plt.show()

# Start measuring the time for T3
start_T3=time.time()
#This is the intialization for the T3
conv2d_T3=Conv2D(in_channel=3,out_channel=3,kernel_size=3,stride=2,mode='known')
operations_T3,output_T3=conv2d_T3.forward(image)
end_T3=time.time()
print('Number of operations for Task 3:',operations_T3)              # print the number of operations 
print ('Output image tensor size for Task 3: ', output_T3.shape)     # Print the output image dimensions 
print('Time (s) to execute the forward pass for Task 3:',end_T3-start_T3)   # Print the processing time
# Indicate the current state of code
print('test.py execution is complete, please close all the open pictures\n')
# Start subplotting the gray output images. Number of channels in output image vary. 
#This will throw an error if the output image channels are less than 3
plt.imshow(output_T3[0,...],cmap=plt.cm.gray),plt.show()
plt.imshow(output_T3[1,...],cmap=plt.cm.gray),plt.show()
plt.imshow(output_T3[2,...],cmap=plt.cm.gray),plt.show()


# Save the different gray scale channels of output image, similar to image subplotting
# Can throw error depeding on the number of channels in output image 
imsave('Image_2_K1_T1.jpg', output_T1[0,...])
imsave('Image_2_K1_T3.jpg', output_T3[0,...])
imsave('Image_2_K2_T3.jpg', output_T3[1,...])
imsave('Image_2_K3_T3.jpg', output_T3[2,...])
imsave('Image_2_K4_T2.jpg', output_T2[0,...])
imsave('Image_2_K5_T2.jpg', output_T2[1,...])