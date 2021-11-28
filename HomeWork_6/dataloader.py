#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:06:04 2018

@author: tanzeel
"""

import shutil,os


class datasetup:
    def __init__(self):
        self=self
    def classes(self,inp_directory):
        # Create an empty 1000 class dictionary whose keys are the alphanumeric names and values are class labels
        thousand_class_dict={}
        small_class_dict={}     # Dictionary for 200 classes
        with open(os.path.join(inp_directory,'words.txt'),'r') as class_list:
            for line in class_list:
                words=line.split('\t')
                label=words[1].split(',')
                thousand_class_dict[words[0]]=label[0]
        #Read the folder names from train directory and compare these names with keys 
        dirct=os.path.join(inp_directory,'train')
        small_folder_names=[f for f in sorted(os.listdir(dirct))]
        for small_labels in small_folder_names:
            for foldernames, label in thousand_class_dict.items():
                if small_labels==foldernames:
                    small_class_dict[foldernames]=label
        return small_class_dict
        
    #Create the Train like folder structure for the Val data
    def val_classes(self,inp_directory):
        val_img_dict={}     # Dictionary for storing all the images as key and folder information as value
        with open(os.path.join(inp_directory,'val/val_annotations.txt'),'r') as annonation:
            for line in annonation:
                words=line.split('\t')
                val_img_dict[words[0]]=words[1]
        val_path=os.path.join(inp_directory,'val/images')
        for images, folder_names in val_img_dict.items():
            folders_path=os.path.join(val_path,folder_names)
            if not os.path.exists(folders_path):
                os.makedirs(folders_path)
            # Combine the val_path and key to read an image
            images=os.path.join(val_path,images)
            if os.path.exists(images):
                shutil.move(images,os.path.join(folders_path,images))

#root='/home/tanzeel/BME495/Tiny-tiny-imagenet-2'
#d=datasetup()
#cl=d.classes(root)
#d.val_classes(root)   