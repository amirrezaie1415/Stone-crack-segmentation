import glob

import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io
from sliding_window import sliding_window, zero_pad
from skimage.morphology import skeletonize

# core = 'Core3'
# path = '../../dataset/original_dataset/' + core
# path_to_save_images = '../../dataset/treated/' + core + os.sep + 'images'
# path_to_save_masks = '../../dataset/treated/' + core + os.sep + 'masks'
#
# os.makedirs(path_to_save_images, exist_ok=True)
# os.makedirs(path_to_save_masks, exist_ok=True)
#
#
# image_list = glob.glob(os.path.join(path, '*.tif'))
# mask_list = glob.glob(os.path.join(path, '*.png'))
#
# for mask in mask_list:
#     os.rename(mask, os.path.join(path_to_save_masks, "full_size", os.path.basename(mask)))
#
# for mask in mask_list:
#     masks_stem = os.path.basename(mask).split('_')[0]
#     src = os.path.join(path, masks_stem + '.tif')
#     dst = os.path.join(path_to_save_images, "full_size", masks_stem + '.tif')
#     os.rename(src, dst)
#


### make masks binary


#################


### sliding window
# core = 'Core0'
# images_path = '../../dataset/treated/' + core + '/images/full_size'
# masks_path = '../../dataset/treated/' + core + '/masks/full_size'
#
# img_patches_path = '../../dataset/treated/' + core + '/images/patches_256'
# mask_patches_path = '../../dataset/treated/' + core + '/masks/patches_full_annot_256'
# desired_size = 256
# image_list = glob.glob(os.path.join(images_path, '*.tif'))
# masks_list = glob.glob(os.path.join(masks_path, '*.png'))
# image_list.sort()
# masks_list.sort()
#
# for ind in range(len(image_list)):
#     img_stem = os.path.basename(image_list[ind]).split('.')[0]
#     img = skimage.io.imread(image_list[ind])
#     mask = skimage.io.imread(masks_list[ind])
#
#     padded_img = zero_pad(img, desired_size=desired_size)
#     padded_mask = zero_pad(mask, desired_size=desired_size)
#
#     img_windows = []
#     xys = []
#     for (x, y, window) in sliding_window(padded_img, desired_size, windowSize=(desired_size, desired_size)):
#         img_windows.append(window)
#         xys.append((x, y))
#     mask_windows = []
#     for (x, y, window) in sliding_window(padded_mask, desired_size, windowSize=(desired_size, desired_size)):
#         mask_windows.append(window)
#
#     for mask_window, img_window, xy in zip(mask_windows, img_windows, xys):
#         if len(np.where(mask_window == 255)[0]) > 50:
#             save_dir = os.path.join(img_patches_path,
#                                     img_stem + "_{:d}".format(xy[0]) + "_{:d}".format(xy[1]) + '.png')
#             skimage.io.imsave(save_dir, img_window)
#             save_dir = os.path.join(mask_patches_path,
#                                     img_stem + "_{:d}".format(xy[0]) + "_{:d}".format(xy[1]) + '_mask.png')
#             skimage.io.imsave(save_dir, mask_window)



### skletonize masks

core = 'Core3'
mask_patches_fullannot_path = '../../dataset/treated/' + core + '/masks/patches_full_annot_256'
mask_patches_skleton_path = '../../dataset/treated/' + core + '/masks/patches_skleton_256'

masks_list = glob.glob(os.path.join(mask_patches_fullannot_path, '*.png'))
masks_list.sort()

for ind in range(len(masks_list)):
    mask = skimage.io.imread(masks_list[ind])
    skleton = skeletonize(mask/255)
    skleton = skleton.astype(np.uint8) * 255

    img_name = os.path.basename(masks_list[ind])
    save_dir = os.path.join(mask_patches_skleton_path, img_name)
    skimage.io.imsave(save_dir, skleton)

