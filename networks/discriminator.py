import sys
sys.path.append('..')

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.utils import SpectralNorm

class NLayerDiscriminator(nn.Module):
    # change cur_dim 
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        use_bies = False
        kernel_size = 3
        padding = 1
        sequence = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding)),
            nn.LeakyReLU(0.2, True)
        ]

        curr_dim = ndf
        for n in range(1, n_layers):
            sequence.extend([
                # Use spectral normalization
                SpectralNorm(nn.Conv2d(curr_dim, min(curr_dim*2, ndf*8), kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ])
            curr_dim *= min(curr_dim*2, ndf*8)

        sequence.extend([
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(curr_dim, min(curr_dim*2, ndf*8), kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ])
        curr_dim *= min(curr_dim*2, ndf*8)

        # Use spectral normalization
        sequence.extend([SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, stride=1, padding=padding))])

        if use_sigmoid:
            sequence.extend([nn.Sigmoid()])

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)



