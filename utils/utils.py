import cv2
import sys
sys.path.append('..')

import os
import logging
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from network_utils.network_utils import Fashion_swapper_dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from scipy.io import loadmat
from CONSTS import MASKPATH, IMAGEPATH


def createswapper_loader(image_path, mask_path, object_one=31, object_two=40, new_size=(300,200)):
    '''
    return two loader
    - for object_one in CCP dataset
    - for object_two in CCP dataset
    '''
    trans = transforms.Compose([transforms.Resize(new_size, 2), transforms.ToTensor()])
    first_object = load_specific_image(IMAGEPATH, MASKPATH, objects=[object_one, object_two])
    
    dataset_one = Fashion_swapper_dataset(first_object, object_one, transform=trans)
    dataset_second = Fashion_swapper_dataset(first_object, object_two, transform=trans)
    loader_one = DataLoader(dataset_one, batch_size=8, shuffle=True, drop_last=True)
    loader_second = DataLoader(dataset_second, batch_size=8, shuffle=True, drop_last=True)
    return loader_one, loader_second

def read_image(name):
    '''
    name - name of image without format
    '''
    try:
        img = Image.open(f'{IMAGEPATH}/{name}.jpg').convert('RGB')
    except IOError as ex:
        logging.exception(str(ex))
        img = None
    return img

def read_mask(name, objects=[]):
    '''
    name - name of image without format
    objects - index of fashion wear in CCP dataset
    '''
    try:
        mask_dic = loadmat(f'{MASKPATH}/{name}')
    except FileNotFoundError as ex:
        logging.exception(str(ex))
        mask_dic = {}
    mask_numpy = mask_dic.get('groundtruth')
    if (mask_numpy is not None):
        mask_bool = np.isin(mask_numpy, objects) if objects else np.isin(mask_numpy, objects, invert=True)
    else:
        mask_numpy = np.array([])
    # np.putmask(mask_numpy, ~mask_bool, 0)
    return np.where(mask_bool, mask_numpy, 0)

def specific_mask(mask, objects=[]):
    return np.isin(mask, objects)

def check_array(image):
    try:
        if not isinstance(image, np.ndarray):
            return np.array(image)
        else:
            return image
    except:
        logging.exception("Can't convert to np.array")

def cut_mask(image, mask, element):
    image = check_array(image)
    cloth_mask = np.where(mask == element, mask, 0)
    image_mask = cv2.bitwise_and(image, image, mask = cloth_mask)
    return image_mask

def add_mask(image, mask, axis=0):
    '''
    add mask to image
    '''
    return torch.tensor(np.concatenate((image, mask), axis=axis))

def load_specific_image(image_path, mask_path, objects=[31, 40]):
    """
    load specific object from dataset
    """
    return_info = {
        'objects' : {},
        'objects_count': {}
    }
    masks = os.listdir(MASKPATH)
    for mask in tqdm(masks):
        mask_array = read_mask(mask)
        classes = np.unique(mask_array)
        image_name = f'{mask.split(".")[0]}'
        image_array = read_image(image_name)
        for object_ in objects:
            if object_ in classes:
                if object_ not in return_info['objects_count']:
                    return_info['objects_count'][object_] = 0
                if object_ not in return_info['objects']:
                    return_info['objects'][object_] = []
                    
                return_info['objects_count'][object_] += 1
                return_info['objects'][object_].append(image_name.split('.')[0])
    return return_info

def kaggle_to_rle_format(arr, height, width):
    """Converts from Kaggle format to COCO RLE format.

    Args:
      arr: segmentation info about one object from 'EncodedPixels' column
      height: height of image
      width: width of image

    Returns:
      Segmentation information about one object in COCO RLE format.
    """
    correct = [0] * (len(arr) + 1)
    curr = 0
    for i in range(len(arr)):
        if i % 2 == 0:
            correct[i] = arr[i] - curr
            curr = arr[i]
        else:
            correct[i] = arr[i]
            curr += arr[i]

    correct[len(arr)] = height * width - curr
    return correct

def rle_to_binary_format(rle, height, width):
    """Converts from COCO RLE to binary mask.

    Args:
      rle: segmentation info about one object in COCO RLE format
      height: height of image
      width: width of image

    Returns:
      Binary mask with information about one object.
    """
    mask = np.zeros((height * width), dtype=np.uint8)
    curr = 0
    for i, item in enumerate(rle):
        mask[curr:curr+int(item)] = 0 if i % 2 == 0 else 1
        curr += int(item)

    mask = np.transpose(np.reshape(mask, (width, height)))
    return mask

def kaggle_to_binary_format(arr, height, width):
    """Converts from Kaggle format to binary mask.

    Args:
      arr: segmentation info about one object from 'EncodedPixels' column
      height: height of image
      width: width of image

    Returns:
      Binary mask with information about one object.
    """
    rle = kaggle_to_rle_format(arr, height, width)
    return rle_to_binary_format(rle, height, width)
