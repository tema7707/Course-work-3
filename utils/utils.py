import cv2
import sys
sys.path.append('..')

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image
from scipy.io import loadmat
from CONSTS import MASKPATH, IMAGEPATH

def read_image(name):
    try:
        img = Image.open(f'{IMAGEPATH}/{name}').convert('RGB')
    except IOError:
        img = None
    return img

def read_mask(name):
    try:
        mask_dic = loadmat(f'{MASKPATH}/{name}')
    except FileNotFoundError:
        mask_dic = {}
    return mask_dic.get('groundtruth')

def cut_mask(image, mask, element):
    cloth_mask = np.where(mask == element, mask, 0)
    image_mask = cv2.bitwise_and(image, image, mask = cloth_mask)
    return image_mask

def kaggle_to_rle_format(arr: List[int], height: int, width: int) -> List[int]:
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

def rle_to_binary_format(rle: List[int], height: int, width: int) -> ndarray:
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

def kaggle_to_binary_format(arr: List[int], height: int, width: int) -> ndarray:
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
