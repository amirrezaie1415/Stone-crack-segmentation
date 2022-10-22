#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:10:22 2021

@author: pantoja
"""

from torch import optim
from evaluation import *
from networks import TernausNet16
import csv
import torch

import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import skimage.io
from utils import zero_pad
from utils import sliding_window
import glob
import pathlib
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")

#Testing
data_folder = 'test/'
#images_path = data_folder+'images'
images_path = "../dataset/test"
# load model
model_name = 'TernausNet16-DiceLoss-256-1-1-1-100-1-0.000200-0.90-0.9990-0.0000000000-0.50-0.pkl'
#model_name = 'dice_or_lb_full_weights_100_1e-05.pt'
threshold = 0.5
model_path = os.path.join('../models', model_name)
result_path = images_path

model = TernausNet16()
#model = UNet16()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location="cuda"))
model.to(device)
model.train(False)
model.eval()

desired_size = 256
transform = []
transform.append(T.Resize((desired_size, desired_size)))
transform.append(T.ToTensor())
transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
transform = T.Compose(transform)

image_names=[i.split(os.sep)[-1] for i in glob.glob(os.path.join(images_path, '*'))]
FILE_EXTENSIONS = ['.jpg', '.JPG',
                   '.jpeg', '.JPEG',
                   '.png', '.PNG',
                   '.tif', '.tiff', '.TIFF',
                   '.bmp', '.BMP']

for ind, file in enumerate(image_names):
    tmp = pathlib.Path(file)
    if tmp.suffix not in FILE_EXTENSIONS:
        del image_names[ind]

for image_name in tqdm(image_names):
    image_file = skimage.io.imread(os.path.join(images_path, image_name))
    org_im_h = image_file.shape[0]
    org_im_w = image_file.shape[1]
    padded_image = zero_pad(image_file, desired_size)

    window_names = []
    windows = [] # as Tensor (ready for to use for deep learning)
    for (x, y, window) in sliding_window(padded_image, step_size=desired_size, window_size=(desired_size, desired_size)):
            window_names.append(image_name[:-4]+ "_{:d}".format(x) + "_{:d}".format(y))
            window = T.ToPILImage()(window) # as PIL
            window = window.convert('RGB')
            window = transform(window)
            windows.append(torch.reshape(window, [1, 3, desired_size, desired_size]))


    overlay_crack = zero_pad(np.zeros((org_im_h, org_im_w), dtype = "uint8"),
                         desired_size = desired_size)

    with torch.no_grad():
        for window, window_name in zip(windows, window_names):
            window = window.to(device)
            SR = model(window)
            SR_probs = torch.sigmoid(SR)
            SR_probs_arr = SR_probs.view(desired_size,desired_size)
            #SR_probs.detach().numpy().reshape(desired_size, desired_size)
            binary_result = SR_probs_arr > threshold
            binary_result = binary_result.to('cpu').detach().numpy()
            y = int(window_name.split('_')[-1])
            x = int(window_name.split('_')[-2])
            overlay_crack[y:y + desired_size, x:x + desired_size] = binary_result
    overlay_crack = overlay_crack[:org_im_h, :org_im_w] * 255

    image_name_save = image_name[:-4] + '_mask.png'
    skimage.io.imsave(os.path.join(result_path, image_name_save), overlay_crack)
    skimage.io.imsave(os.path.join(result_path, image_name), image_file)

    overlay_name_save = image_name[:-4] + '_overlay.jpg'
    prediction_rgb = np.zeros((overlay_crack.shape[0], overlay_crack.shape[1], 3), dtype='uint8')
    prediction_rgb[:,:,0] = overlay_crack

    # overlayed_prediction = image_file
    # overlayed_prediction[np.where(overlay_crack == 255)[0][0], np.where(overlay_crack==255)[0][1], 0] = 255
    # overlayed_prediction[np.where(overlay_crack == 255)[0][0], np.where(overlay_crack == 255)[0][1],1:] = 0
    if np.ndim(image_file) == 2:
        image_file = np.stack((image_file,)*3, axis=-1)
    overlayed_prediction = cv.addWeighted(image_file, 1.0, prediction_rgb, 1.0, 0)
    skimage.io.imsave(os.path.join(result_path, overlay_name_save), overlayed_prediction)
