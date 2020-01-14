import os
import sys
sys.path.append('..')

import torch
from PIL import Image
import numpy as np
from utils.utils import read_image, read_mask, cut_mask
from CONSTS import IMAGEPATH, MASKPATH

class FashionDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root 
        self.transforms = transforms 

        self.imgs = list(sorted(os.listdir(os.path.join(root, 'photos'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'pixel_level'))))

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, 'photos', self.imgs[idx])
        mask_path = os.path.join(self.root, 'pixel_level', self.masks[idx])
        img = read_image(self.imgs[idx])
        mask = read_mask(self.masks[idx])

        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target