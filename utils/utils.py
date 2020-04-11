import cv2
import sys
sys.path.append('..')

import os
import logging
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from scipy.io import loadmat
from CONSTS import MASKPATH, IMAGEPATH

class Visualize:
    '''
    Example:
    >>vis = Visualize()

    >>x = np.arange(-10, 10)
    y_train = lambda x: np.sin(x)
    y_val = lambda x: np.cos(x)
    for xi in x:
        plt.figure(figsize=(10, 10))
        vis.history['train'].append(y_train(xi))
        vis.history['val'].append(y_val(xi))
        vis.plot_loss_curve()
    '''
    def __init__(self):
        self.history = {
            'train':[], 
            'val':[],
            }
    def plot_loss_curve(self):
        length = np.array([i for i in range(len(self.history['train']))]).reshape(-1)
        plt.plot(length, self.history['train'], label='train')
        plt.plot(length, self.history['val'], label='val')
        clear_output(wait=True)
        plt.legend()
        plt.show()

class Fashion_swapper_dataset(Dataset):
    
    def __init__(self, loader, objects=[31, 40], transform=None):
        self.objects = objects
        self.transform = transform
        self.loader = loader
        self.len_first_obj = loader['objects_count'][self.objects[0]]
        self.len_second_obj = loader['objects_count'][self.objects[1]]
        self.iterator = []
        for i in range(self.len_first_obj):
          for j in range(self.len_second_obj):
            self.iterator.append((i, j))
        
    def __getitem__(self, idx):        
        first_name = self.loader['objects'][self.objects[0]][self.iterator[idx][0]]
        first_image = read_image(first_name)
        
        first_mask = read_mask(first_name, self.objects[0])
        first_mask = np.where(first_mask == 0, first_mask, 1.)
        first_mask = Image.fromarray(first_mask)
        first_image = self.transform(first_image)
        first_mask = self.transform(first_mask)
        imagewithmask_first = add_mask(first_image, first_mask, axis=0)
        
        second_name = self.loader['objects'][self.objects[1]][self.iterator[idx][1]]
        second_image = read_image(second_name)
        
        second_mask = read_mask(first_name, self.objects[1])
        second_mask = np.where(second_mask == 0, second_mask, 1.)
        second_mask = Image.fromarray(second_mask)
        second_image = self.transform(second_image)
        second_mask = self.transform(second_mask)
        imagewithmask_second = add_mask(second_image, second_mask, axis=0)
        
        return imagewithmask_first, imagewithmask_second
    
    def __len__(self):
        return self.len_first_obj * self.len_second_obj

class CPDataset(Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt, stage='GMM', data_list='train_pairs.txt', datamode='train'):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.datamode = datamode # train or test or self-defined
        self.stage = "GMM" # GMM or TOM
        self.data_list = data_list
        self.fine_height = 192
        self.fine_width = 256
        self.radius = 3
        self.data_path = os.path.join(opt, self.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_body = transforms.Compose([transforms.ToTensor()])
        
        # load data list
        im_names = []
        c_names = []
        with open(os.path.join(opt, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        if self.stage == 'GMM':
            c = Image.open(os.path.join(self.data_path, 'cloth', c_name))
            cm = Image.open(os.path.join(self.data_path, 'cloth-mask', c_name))
        else:
            c = Image.open(os.path.join(self.data_path, 'warp-cloth', c_name))
            cm = Image.open(os.path.join(self.data_path, 'warp-mask', c_name))
     
        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0)

        # person image 
        im = Image.open(os.path.join(self.data_path, 'image', im_name))
        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(os.path.join(self.data_path, 'image-parse', parse_name))
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)
        # parse_shape = parse_shape[:,:,np.newaxis]
        # print(np.array(parse_shape).shape)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
       
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_height//16, self.fine_width//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_height, self.fine_width), Image.BILINEAR)
        shape = self.transform_body(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointy-r, pointx-r, pointy+r, pointx+r), 'white', 'white')
                pose_draw.rectangle((pointy-r, pointx-r, pointy+r, pointx+r), 'white', 'white')
            one_map = self.transform_body(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform_body(im_pose)
        im_pose = torch.tensor(np.array(im_pose).transpose(0,2,1))
        
        # cloth-agnostic representation
        # print(shape.shape)
        # print(im_h.shape)
        pose_map = torch.tensor(np.array(pose_map).transpose(0,2,1))
        agnostic = torch.cat([shape, im_h, pose_map], 0) 

        # if self.stage == 'GMM':
        #     im_g = Image.open('grid.png')
        #     im_g = self.transform(im_g)
        # else:
        #     im_g = ''

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            # 'grid_image': im_g,     # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset, shuffle=True, batch_size=4):
        super(CPDataLoader, self).__init__()

        if shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


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

def convert_image(image, hight_res=256, weight_res=192, conv_type='resize'):
    '''
    image: numpy.array
    conv_type: string, 'crop'/'resize'
    '''
    hight, weight, _ = image.shape
    coef = hight_res / weight_res
    new_x = weight
    new_y = hight
    if hight / weight < coef:
        new_x = int(hight / coef) 
    if hight / weight > coef:
        new_y = int(weight * coef)
    image = image[hight//2-new_y//2:hight//2+new_y//2, weight//2-new_x//2:weight//2+new_x//2, :]
    if conv_type == 'crop':
        startx = weight//2-(weight_res//2)
        starty = hight//2-(hight_res//2)    
        return image[starty:starty+hight_res,startx:startx+weight_res]
    if conv_type == 'resize':
        return cv2.resize(image, (weight_res, hight_res), interpolation = cv2.INTER_AREA)

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
