import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):

        super(ResidualBlock, self).__init__()
        self._name = 'residual_block'
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel, affine=True))

    def forward(self, x):
        return x + self.res_block(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1):

        super(DownsamplingBlock, self).__init__()
        self._name = 'downsampling_block'

        self.in_channel, self.out_channel = in_channel, out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.downsampling_block = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
            nn.InstanceNorm2d(self.out_channel, affine=True),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.downsampling_block(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1):

        super(UpsamplingBlock, self).__init__()
        self._name = 'upsampling_block'

        self.in_channel, self.out_channel = in_channel, out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.upsampling_block = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel, self.out_channel, kernel_size=self.kernel_size, stride=self.stride, 
                            padding=self.padding, output_padding=self.output_padding, bias=False),
            nn.InstanceNorm2d(self.out_channel, affine=True),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.upsampling_block(x)


class ResNetGenerator_encoder(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, kernel_size=4, n_down=2):
        super(ResNetGenerator_encoder, self).__init__()
        self._name = 'resnetgenerator_encoder'
        layers = []
        layers.append(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.curr_dim = conv_dim
        for i in range(n_down):
            layers.append(DownsamplingBlock(self.curr_dim, self.curr_dim*2, kernel_size=kernel_size, stride=2, padding=1))
            self.curr_dim = self.curr_dim * 2
        for i in range(repeat_num):
            layers.append(ResidualBlock(in_channel=self.curr_dim, out_channel=self.curr_dim))

        self.model = nn.Sequential(*layers)

    @property
    def _get_curr_dim(self):
        return self.curr_dim

    def forward(self, x):
        # maybe need some preprocessing 
        return self.model(x)


class ResNetGenerator_decoder(nn.Module):
    def __init__(self, curr_dim=5, kernel_size=4, n_down=2):
        super(ResNetGenerator_decoder, self).__init__()
        self._name = 'resnetgenerator_encoder'
        layers = []

        self.curr_dim = curr_dim
        for i in range(n_down):
            layers.append(UpsamplingBlock(self.curr_dim, self.curr_dim//2, kernel_size=kernel_size, stride=2, padding=1))
            self.curr_dim = self.curr_dim // 2

        # layers.append(nn.Conv2d(self.curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        # layers.append(nn.Conv2d(self.curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Conv2d(self.curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    @property
    def _get_curr_dim(self):
        return self.curr_dim

    def forward(self, x):
        # todo: maybe need some preprocessing 
        return self.model(x)


class ResNetGenerator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, kernel_size=3, n_down=2):
        # todo: split parametrs for encoder and decoder 
        super(ResNetGenerator, self).__init__()
        self._name = 'resnet_generator'

        self.curr_dim = c_dim

        self.encoder = ResNetGenerator_encoder(conv_dim=conv_dim, c_dim=self.curr_dim, repeat_num=repeat_num, kernel_size=kernel_size, n_down=n_down)
        self.curr_dim = self.encoder._get_curr_dim
        self.decoder = ResNetGenerator_decoder(curr_dim=self.curr_dim, kernel_size=kernel_size, n_down=n_down)
        self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x):
        return self.model(x)
        # return self.encoder(x)





