import cv2
import sys
sys.path.append('..')

import numpy as np
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
