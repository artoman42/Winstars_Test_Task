"""Script with functions for working with images and rle encoding. """

import numpy as np
import sys
import os
import json
import keras.backend as K
from scipy import ndimage

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

CONF_FILE = "settings.json"

# Load configuration settings from JSON
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)


def crop3x3(img, i):
    """img: np.ndarray - original image 768x768
       i: int 0-8 - image index from crop: 0 1 2
                                           3 4 5
                                           6 7 8
       returns: image 256x256 
    """
    return img[(i//3)*conf['general']['cropped_image_size']: ((i//3)+1)*conf['general']['cropped_image_size'],
               (i%3)*conf['general']['cropped_image_size']: (i%3+1)*conf['general']['cropped_image_size']]


def crop3x3_mask(img):
    """Returns crop image, crop index with maximum ships area"""
    i = K.argmax((
        K.sum(crop3x3(img, 0)),
        K.sum(crop3x3(img, 1)),
        K.sum(crop3x3(img, 2)),
        K.sum(crop3x3(img, 3)),
        K.sum(crop3x3(img, 4)),
        K.sum(crop3x3(img, 5)),
        K.sum(crop3x3(img, 6)),
        K.sum(crop3x3(img, 7)),
        K.sum(crop3x3(img, 8)),
    ))
    return (crop3x3(img, i), i)

def rle_decode(mask_rle, input_shape=(768,768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    img=np.zeros(input_shape[0]*input_shape[1], dtype=np.float32)
    if not(type(mask_rle) is float):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1.0
    return img.reshape((input_shape[0],input_shape[1])).T

def show_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype = np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += show_decode(mask)
    return np.expand_dims(all_masks, -1)

def split_mask(mask, threshold = 0.6,  threshold_obj = 8  ):
        """
        Split the input mask into individual objects.

        Args:
            mask (numpy.ndarray): Binary mask representing objects, where values above a threshold are considered part of an object.
            threshold (float): Threshold value for considering a pixel as part of an object. Defaults to 0.6.
            threshold_obj (int): Minimum number of pixels for an object to be considered. Defaults to 8.

        Returns:
            list: A list of numpy arrays, each representing a segmented object from the input mask.
        """
        labeled,n_objs = ndimage.label(mask > threshold)
        result = []
        for i in range(n_objs):
            obj = (labeled == i + 1).astype(int)
            if(obj.sum() > threshold_obj): result.append(obj)
        return result

def rle_encode(img):
    """fucntion to make rle encoding of mask"""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_run_length_encoded_predictions(y_pred, img_name):
    """function to get rle encoded predictions"""
    list_dict = []
    masks = split_mask(y_pred)
    if len(masks) == 0:
        list_dict.append({"ImageId": img_name, "EncodedPixels": np.nan})
    for mask in masks:
        list_dict.append({"ImageId": img_name, "EncodedPixels": rle_encode(mask)})
    return list_dict
