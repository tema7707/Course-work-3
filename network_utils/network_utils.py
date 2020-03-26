import sys
sys.path.append('..')
sys.path.append('.')

import torch.utils.data
import torch
import os
from torchvision import transforms
import numpy as np
import json

import torch.nn as nn

from PIL import Image, ImageDraw
from torch.utils.data import Dataset
# from utils.utils import read_image, read_mask, add_mask

class CPDataset(Dataset):
    def __init__(self, dataroot='data_viton', datamode='train', stage='GMM', data_list='train_pairs.txt'):
        super(CPDataset, self).__init__()
        self.root = dataroot
        self.datamode = datamode
        self.stage = stage
        self.data_list = data_list
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5
        self.data_path = os.path.join(dataroot, datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                
        self.transform_1d = transforms.Compose([ \
                transforms.ToTensor(), \
                transforms.Normalize((0.5,), (0.5,))])

        self.transform_1d_x = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5), (0.5))])

        im_names = []
        c_names = []
        with open(os.path.join(dataroot, data_list), 'r') as f:
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
        #$$ im_parse.shape = [1,256,192]. png color image is with colormap.
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32) #$$ 0 or 1
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
       
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8)) #$$ 0 or 255
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR) #$$ why
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        # – Body shape: a 1-channel feature map of a blurred binary mask that roughly covering different parts of human body.
        shape = self.transform_1d(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts #$$ not good.

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
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform_1d(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform_1d(im_pose)
        
        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0)  #$$ 1 + 1? + 18 =

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
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def l2normalize(self, vector, eps = 1e-15):
        return vector/(vector.norm()+eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = torch.tensor(1.0)
        else:
            target_tensor = torch.tensor(0.0)
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)