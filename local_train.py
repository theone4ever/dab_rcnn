import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model_50 as modellib
import visualize
from model import log
from dsb2018 import Dsb2018Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



# %matplotlib inline

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Root directory of the project
ROOT_DIR = os.getcwd()


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "dsb2018_logs_res50")



# Training dataset
train_data = Dsb2018Dataset()
train_data.load_dsb2018('../DSB2018')
train_data.prepare()

val_data = Dsb2018Dataset()
val_data.load_dsb2018('../DSB2018', is_trainset=False)
val_data.prepare()
print("Size of train data:{}".format(len(train_data.image_ids)))
print("Size of val data:{}".format(len(val_data.image_ids)))


# #Load and display random samples
# image_ids = np.random.choice(train_data.image_ids, 2)
# for image_id in image_ids:
#     image = train_data.load_image(image_id)
#     mask, class_ids = train_data.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, train_data.class_names, limit=1)
#
#
# #Load and display random samples
# image_ids = np.random.choice(val_data.image_ids, 2)
# for image_id in image_ids:
#     image = val_data.load_image(image_id)
#     mask, class_ids = val_data.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, val_data.class_names, limit=1)

from skimage.io import imread, imshow, imread_collection, concatenate_images

from keras.preprocessing.image import ImageDataGenerator

def argument_img_mask(image, mask, class_ids):
    common_seed = 7

    #print("origin image shape:{}, origina mask shape:{}".format(image.shape, mask.shape))

    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1
                         )

    xtr = np.expand_dims(image, 0)
    image_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(xtr, seed=common_seed)
    image_generator = image_datagen.flow(xtr, batch_size=1, seed=common_seed)
    arg_image = image_generator.next()[0].astype(image.dtype)

    arg_mask = np.zeros(mask.shape, dtype=mask.dtype)

    for i in range(mask.shape[-1]):
        mask_datagen = ImageDataGenerator(**data_gen_args)
        masktr = np.expand_dims(mask[:,:,i], 0)
        masktr = np.expand_dims(masktr, -1)
        mask_datagen.fit(masktr, seed=common_seed)
        mask_generator = mask_datagen.flow(masktr, batch_size=1, seed=common_seed)
        arg_mask_ = np.squeeze(mask_generator.next()[0], axis=-1)
        # print("arg_mask_ shape:{}".format(arg_mask_.shape))
        arg_mask[:,:,i] = arg_mask_.astype(mask[:,:,i].dtype)

    # remove the mask instance which doesn't contain any mask after argumentation
    non_zero_mask = arg_mask[:,:, ~np.all(arg_mask == 0, axis=(0, 1))]

    class_ids = class_ids[:non_zero_mask.shape[-1]]
    #print("arg_mask shape:{}, non_zero_mask shape:{}, class_ids shape:{}".format(arg_mask.shape, non_zero_mask.shape, class_ids.shape))


    # print("arg_mask shape:{}".format(arg_mask.shape))

    return (arg_image,non_zero_mask, class_ids)


image_ids = np.random.choice(train_data.image_ids, 1)
for image_id in image_ids:
    image = train_data.load_image(image_id)
    mask, class_ids = train_data.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, train_data.class_names, limit=1)
    output_image, output_masks = modellib.data_augmentation(image, mask )
    visualize.display_top_masks(output_image, output_masks, class_ids, train_data.class_names, limit=1)
