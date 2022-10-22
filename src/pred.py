"""
This file is used to predict test data.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import get_loader
from networks import *
import torch
import skimage.io
from evaluation import *
import csv

# input from the user
index_model = 0
architecture = 'TernausNet16'
image_size = 256
test_data_with_ground_truth = False  # if there is not any ground-truth for the test data,
# this variable must set to False.
images_group = 'test'
save_pred = False

# model path
model_path = '../models/'
model_name = 'TernausNet16-DiceLoss-256-1-1-1-100-1-0.000200-0.90-0.9990-0.0000000000-0.50-0.pkl'
model_path = os.path.join(model_path, model_name)


# images path
images_path = '../dataset/' + images_group + '/'
imageNames = []
for imageFile in os.listdir(images_path):
    if imageFile[-4:] in ['.JPG', '.tif', '.png']:
        imageNames.append(imageFile[:-4])

# predictions path
predictions_path = '../dataset/' + images_group + '/'

if test_data_with_ground_truth:
    test_loader = get_loader(image_path=images_path,
                             image_size=image_size,
                             batch_size=1,
                             num_workers=0,
                             mode='valid',
                             augmentation_prob=0.,
                             shuffle_flag=False)

    # load a trained model
    model = vars()[architecture]()
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model.eval()


    with torch.no_grad():
        # save side by side results
        for idx, (image, GT) in enumerate(test_loader):
            SR = model(image)
            SR_probs = torch.sigmoid(SR)
            SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
            GT_arr = GT.detach().numpy().reshape(image_size, image_size)
            plt.figure(figsize=(5,5))
            plt.imshow(SR.detach().numpy().reshape(image_size, image_size))
            plt.show()
            binary_result = SR_probs_arr > 0.5
            image_numpy = image.detach().numpy()
            image_numpy = image_numpy[0, 0, :, :]
            fig, ax = plt.subplots(figsize=(12, 16), nrows=1, ncols=3)
            ax[0].imshow(image_numpy, cmap=plt.cm.gray)
            ax[0].set_title('Image')
            ax[0].axis('off')
            ax[1].imshow(GT_arr, cmap=plt.cm.binary_r)
            ax[1].set_title('Ground truth')
            ax[1].axis('off')
            ax[2].imshow(binary_result, cmap=plt.cm.flag_r)
            ax[2].set_title('Prediction')
            ax[2].axis('off')
            plt.show()
            binary_RGB = np.zeros((binary_result.shape[0], binary_result.shape[1],3))
            binary_RGB[:,:,0] = binary_result * 255

            if save_pred:
                save_dir = os.path.join(predictions_path, imageNames[idx] + '_mask.png')
                skimage.io.imsave(save_dir, np.array(binary_result, dtype='uint8') * 255)
                save_dir = os.path.join(predictions_path, imageNames[idx] + '_mask_RGB.png')
                skimage.io.imsave(save_dir, np.array(binary_RGB, dtype='uint8'))





else:
    test_loader = get_loader(image_path=images_path,
                             image_size=image_size,
                             batch_size=1,
                             num_workers=0,
                             mode='test',
                             augmentation_prob=0.,
                             shuffle_flag=False)

    # load a trained model
    model = vars()[architecture]()
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model.eval()

    # save side by side results
    for idx, image in enumerate(test_loader):
        SR = model(image)
        SR_probs = torch.sigmoid(SR)
        SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
        binary_result = SR_probs_arr > 0.5
        image_numpy = image.detach().numpy()
        image_numpy = image_numpy[0, 0, :, :]
        fig, ax = plt.subplots(figsize=(12, 16), nrows=1, ncols=2)
        ax[0].imshow(image_numpy, cmap=plt.cm.gray)
        ax[0].set_title('image')
        ax[0].axis('off')
        ax[1].imshow(binary_result, cmap=plt.cm.flag_r)
        ax[1].set_title('prediction')
        ax[1].axis('off')
        plt.show()
        if save_pred:
            save_dir = os.path.join(predictions_path, imageNames[idx] + '_mask.png')
            skimage.io.imsave(save_dir, np.array(binary_result, dtype="uint8") * 255)
