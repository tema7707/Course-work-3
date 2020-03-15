import sys
sys.path.append('..')

import torch
import torch.nn as nn

from networks.discriminator import PatchDiscriminator
from networks.generator import ResNetGenerator, ResNetUnetGenerator

class CycleGAN(nn.Module):
    '''
    Descriminator eat 128x128 image
    '''
    def __init__(self, base_gen_model='resnet'):
        '''
        base_gen_model: string - resnet/resnetunet
        '''
        super(CycleGAN, self).__init__()

        if base_gen_model == 'resnet':
            self.G_A = ResNetGenerator(conv_dim=64, c_dim=4, repeat_num=9, kernel_size=3, n_down=4)
            self.G_B = ResNetGenerator(conv_dim=64, c_dim=4, repeat_num=9, kernel_size=3, n_down=4)
        elif base_gen_model == 'resnetunet':
            self.G_A = ResNetUnetGenerator(conv_dim=64, c_dim=4, repeat_num=9, kernel_size=3, n_down=4)
            self.G_B = ResNetUnetGenerator(conv_dim=64, c_dim=4, repeat_num=9, kernel_size=3, n_down=4)

        self.D_A = PatchDiscriminator(c_dim=4, conv_dim=64, n_layers=3)
        self.D_B = PatchDiscriminator(c_dim=4, conv_dim=64, n_layers=3)
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
        self.criterionCyc = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()


    def forward(self, x):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B) 
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A) 