# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:45:02 2020

@author: rezaie
"""

## Load necessary packages and instances
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
import cv2

def zero_pad(image, desired_size): # imgae : RGB

    remainder_y = image.shape[0] % desired_size
    newImgSize_y = image.shape[0] + desired_size - remainder_y
    remainder_x = image.shape[1] % desired_size
    newImgSize_x = image.shape[1] + desired_size - remainder_x

    if image.ndim == 3:
        new_im = np.zeros((newImgSize_y, newImgSize_x, 3), dtype = "uint8")
        new_im[0:image.shape[0], 0:image.shape[1], :] = image[:,:,:]
        return new_im
    if image.ndim == 2:
        new_im = np.zeros((newImgSize_y, newImgSize_x), dtype = "uint8")
        new_im[0:image.shape[0], 0:image.shape[1]] = image[:,:]
        return new_im 


# Load images
def main_image_from_slices(data_folder,full_image_path, patches_path, type_img = None, windowSize = (256,256)):
    #image_path = os.getcwd()

    #windowSize = (256, 256)
    list_images = os.listdir(full_image_path)
    list_patches = os.listdir(patches_path)

    #Check if directory for full images exists, if not, create it
    check_dir = os.path.isdir('results_damage/'+data_folder+'full/')
    if not check_dir:
        os.makedirs('results_damage/'+data_folder+'full/')

    #images = {}
    for im in list_images:
        imageNames = []
        image_files = []
        #images[im[:-4]] = skimage.io.imread(image_path+im)
        im_name = im[:-4]
        img = skimage.io.imread(full_image_path+im)
        
        #print(img.shape)
        org_im_h = img.shape[0]
        org_im_w  = img.shape[1]
        #n_ch = len(img.shape)
        overlay_crack = zero_pad(img.astype("uint8"), desired_size = windowSize[0])
        
        
        #overlay_crack = zero_pad(np.zeros((org_im_h, org_im_w), dtype = "uint8"), 
        #                     desired_size = windowSize[0])
        
        img_patches = [p for p in list_patches if im_name in p]
        if type_img is not None:
            img_patches = [p for p in img_patches if type_img in p]
        else:
            img_patches = [p for p in img_patches if "pred_bin" not in p] 
            img_patches = [p for p in img_patches if "overlayed" not in p]
            img_patches = [p for p in img_patches if "pred_dm" not in p]
        
        #for imageFile in os.listdir(image_path):
        for imageFile in img_patches:
            if imageFile[-4:] in ['.png']: 
                imageNames.append(imageFile[:-4])
                image_files.append(skimage.io.imread(os.path.join(patches_path, imageFile)))
        
        #Channels number in patches
        n_ch = len(image_files[0].shape)
        #Change chanels in base image
        if n_ch==2:
            overlay_crack = overlay_crack[:,:,0]

        for image_id in range(len(image_files)):
            #print(imageNames[image_id].split('_'))
            if type_img==None:                
                x = int(imageNames[image_id].split('_')[-2])
                y = int(imageNames[image_id].split('_')[-1])
            else:
                x = int(imageNames[image_id].replace("_"+type_img, "").split('_')[-2])
                y = int(imageNames[image_id].replace("_"+type_img, "").split('_')[-1])
            
            #print(x,y)

            overlay_crack[y:y + windowSize[1], x:x + windowSize[0]] = image_files[image_id]

# =============================================================================
#             if n_ch==3:
#                 overlay_crack[y:y + windowSize[1], x:x + windowSize[0],:] = image_files[image_id]
#             else:
#                 overlay_crack[y:y + windowSize[1], x:x + windowSize[0]] = image_files[image_id]
# =============================================================================
        
        overlay_crack = overlay_crack[:org_im_h, :org_im_w]
        #split_name = imageNames[0].split('_')
        
        if type_img is not None:
            #image_name_save = "_".join(split_name[:-4]) + '_mask.png'
            image_name_save = im_name + '_' + type_img + '.png'
        else:
            image_name_save = im_name + '.png'
        
        skimage.io.imsave('results_damage/'+data_folder+'full/' + image_name_save, overlay_crack)
                
