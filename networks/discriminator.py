import sys
sys.path.append('..')

import torch.nn as nn
import torch.nn.functional as F
import torch
from network_utils.network_utils import SpectralNorm


class PatchDiscriminator(nn.Module):
    def __init__(self, c_dim, conv_dim=64, n_layers=3, use_sigmoid=False):
        super(PatchDiscriminator, self).__init__()

        norm_layer = nn.BatchNorm2d
        kernel_size = 4
        padding = 1
        sequence = [nn.Conv2d(c_dim, conv_dim, kernel_size=kernel_size, stride=2, padding=padding), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                SpectralNorm(nn.Conv2d(conv_dim * nf_mult_prev, conv_dim * nf_mult, kernel_size=kernel_size, stride=2, padding=padding, bias=False)),
                norm_layer(conv_dim * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            SpectralNorm(nn.Conv2d(conv_dim * nf_mult_prev, conv_dim * nf_mult, kernel_size=kernel_size, stride=1, padding=padding, bias=False)),
            norm_layer(conv_dim * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(conv_dim * nf_mult, 1, kernel_size=17, stride=1, padding=padding)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)



