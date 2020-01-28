import sys
sys.path.append('..')

import torch
import torch.nn as nn

from networks.discriminator import PatchDiscriminator
from networks.generator import ResNetGenerator

class CycleGAN(nn.Module):
    '''
    Descriminator eat 128x128 image
    '''
    def __init__(self):
        super(CycleGAN, self).__init__()

        self.G_A = ResNetGenerator()
        self.G_B = ResNetGenerator()

        self.D_A = PatchDiscriminator()
        self.D_B = PatchDiscriminator()


    def forward(self, x):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B) 
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A) 