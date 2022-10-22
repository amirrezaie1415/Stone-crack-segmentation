# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:13:46 2019

@author: rezaie
"""

## Load necessary packages and instances
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray
import cv2
from skimage.util import img_as_ubyte
# Load images

image_path = '/home/swissinspect/Projects/malek_crack_detection/dataset/treated/Core0/masks/full_size/'
imageNames = []
image_files = []

for imageFile in os.listdir(image_path):
    if imageFile[-4:] == '.png': 
        imageNames.append(imageFile[:-4])
        image = img_as_ubyte(rgb2gray(skimage.io.imread(os.path.join(image_path, imageFile))))
        image[np.where(image == np.max(image))] = 255
        image = image.astype('uint8')
        image_files.append(image)
        

for index in range(len(image_files)):
    save_dir = os.path.join(image_path, imageNames[index]+ '.png')
    skimage.io.imsave(save_dir, image_files[index])




