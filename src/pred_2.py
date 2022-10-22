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
from tqdm import tqdm

# input from the user
index_model = 0
architecture = 'TernausNet16'
image_size = 256
multi_scale_input=False
test_data_with_ground_truth = False  # if there is not any ground-truth for the test data,
# this variable must set to False.
images_group = 'test'
save_pred = True

dirrectory_names = os.getcwd().split('\\')
root = ''
for ind in range(len(dirrectory_names) - 1):
    root += dirrectory_names[ind] + '/'
# model path
model_path = '../models/'
model_names = []
for file in os.listdir(model_path):
    if file.endswith(".pkl"):
        model_names.append(file)
    
model_path_list = []
for model_ind in range(len(model_names)):
    model_path_list .append(model_path + model_names[model_ind])

# images path
images_path = '../dataset/' + images_group + '/'
imageNames = []
for imageFile in os.listdir(images_path):
    if imageFile[-4:] in ['.JPG', '.tif', '.png']:
        imageNames.append(imageFile[:-4])

# predictions path
predictions_path = '../dataset/' + images_group + '/'

test_loader = get_loader(image_path=images_path,
                         image_size=image_size,
                         batch_size=1,
                         num_workers=0,
                         mode='test',
                         augmentation_prob=0.,
                         shuffle_flag=False)




image_number = 0
seg_avg = torch.zeros((len(model_names), len(test_loader), image_size, image_size))
for model_idx in tqdm(range(len(model_names))):
    # load a trained model
    model = vars()[architecture]()
    model.load_state_dict(torch.load(model_path_list[model_idx]))
    model.train(False)
    model.eval()
    
    with torch.no_grad():
        # save side by side results
        for idx, (image, GT) in tqdm(enumerate(test_loader)):
            image_number += 1
            SR = model(image)
            SR_probs = torch.sigmoid(SR)
            seg_avg[model_idx, idx, :, :] = SR_probs

acc = 0
SE = 0
SP = 0
PC = 0
JS = 0
DC = 0
test_result = {}
seg_avg_2 = torch.mean(seg_avg, dim=0)
for idx, (image, GT) in tqdm(enumerate(test_loader)):
    SR_probs = seg_avg_2[idx,:,:]
    SR_probs = SR_probs.view(1,1,256,256)
    SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
    GT_arr = GT.detach().numpy().reshape(image_size, image_size)
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
    acc += get_accuracy(SR_probs, GT)
    SE += get_sensitivity(SR_probs, GT)
    SP += get_specificity(SR_probs, GT)
    PC += get_precision(SR_probs, GT)
    JS += get_JS(SR_probs, GT)
    DC += get_DC(SR_probs, GT)
    print('DC={}'.format(get_DC(SR_probs, GT)))
    if save_pred:
        save_dir = os.path.join(predictions_path, imageNames[idx] + '_mask.png')
        skimage.io.imsave(save_dir, np.array(binary_result, dtype='uint8') * 255)

image_numbe = 100
test_result['test_acc'] = acc / image_number
test_result['test_SE'] = SE / image_number
test_result['test_SP'] = SP / image_number
test_result['test_PC'] = PC / image_number
test_result['test_JS'] = JS / image_number
test_result['test_DC'] = DC / image_number
print('AC={}, SE={}, SP={}, PC={}, JS={}, DC={}'.format(
    test_result['test_acc'], test_result['test_SE'],
    test_result['test_SP'], test_result['test_PC'],
    test_result['test_JS'], test_result['test_DC']))



