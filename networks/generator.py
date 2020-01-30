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


class Encoder(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, kernel_size=4, n_down=2):
        super(Encoder, self).__init__()
        self._name = 'resnetgenerator_encoder'
        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                                    nn.InstanceNorm2d(conv_dim, affine=True),
                                    nn.ReLU(inplace=True)))
        self.curr_dim = conv_dim
        for i in range(n_down):
            layers.append(DownsamplingBlock(self.curr_dim, self.curr_dim*2, kernel_size=kernel_size, stride=2, padding=1))
            self.curr_dim = self.curr_dim * 2

        self.encoder = nn.Sequential(*layers)

    @property
    def _get_curr_dim(self):
        return self.curr_dim

    def forward(self, x):
        return self.encoder(x)


class Bottleneck(nn.Module):
    def __init__(self, c_dim=256, repeat_num=9):
        super(Bottleneck, self).__init__()
        self.layers = []
        for i in range(repeat_num):
            self.layers.append(ResidualBlock(in_channel=c_dim, out_channel=c_dim))
        self.bottleneck = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.bottleneck(x)


class Decoder(nn.Module):
    def __init__(self, curr_dim=5, kernel_size=4, n_down=2, skip_connection=False):
        super(Decoder, self).__init__()
        self._name = 'resnetgenerator_encoder'
        self.layers = nn.ModuleList()
        self.skip = nn.ModuleList()

        self.curr_dim = curr_dim
        for i in range(n_down):
            self.layers.append(nn.Sequential(
                UpsamplingBlock(self.curr_dim, self.curr_dim//2, kernel_size=kernel_size, stride=2, padding=1),
                nn.InstanceNorm2d(self.curr_dim//2, affine=True),
                nn.ReLU(inplace=True)
            ))
            if skip_connection:
                self.skip.append(nn.Sequential(
                    nn.Conv2d(self.curr_dim, self.curr_dim//2, kernel_size=kernel_size, stride=1, padding=1, bias=False),
                    nn.InstanceNorm2d(self.curr_dim//2, affine=True),
                    nn.ReLU(inplace=True)
                ))
            self.curr_dim = self.curr_dim // 2

        self.layers.append(nn.Sequential(
            nn.Conv2d(self.curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        ))

        self.decoder = nn.Sequential(*self.layers)

    @property
    def _get_curr_dim(self):
        return self.curr_dim

    def forward(self, x): 
        return self.decoder(x)


class ResNetGenerator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, kernel_size=3, n_down=2):
        super(ResNetGenerator, self).__init__()
        self._name = 'resnet_generator'

        self.curr_dim = c_dim

        self.encoder = Encoder(conv_dim=conv_dim, c_dim=self.curr_dim, kernel_size=kernel_size, n_down=n_down)
        self.curr_dim = self.encoder._get_curr_dim
        self.bottleneck = Bottleneck(c_dim=self.curr_dim, repeat_num=repeat_num)
        self.decoder = Decoder(curr_dim=self.curr_dim, kernel_size=kernel_size, n_down=n_down)
        self.model = nn.Sequential(self.encoder, self.decoder) # , self.bottleneck, self.decoder

    def forward(self, x):
        return self.model(x)
        # return self.encoder(x)


class ResNetUnetGenerator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, kernel_size=3, n_down=2):
        super(ResNetUnetGenerator, self).__init__()
        self._name = 'resnet_generator'

        self.curr_dim = c_dim
        self.n_down = n_down

        self.encoder = Encoder(conv_dim=conv_dim, c_dim=self.curr_dim, kernel_size=kernel_size, n_down=self.n_down)
        self.curr_dim = self.encoder._get_curr_dim
        self.bottleneck = Bottleneck(c_dim=self.curr_dim, repeat_num=repeat_num)
        self.decoder = Decoder(curr_dim=self.curr_dim, kernel_size=kernel_size, n_down=n_down, skip_connection=True)

    def forward(self, x):
        
        e_out = self.encoder.encoder[0](x)

        e_result = [e_out]

        for i in range(1, self.n_down + 1):
            e_out = self.encoder.encoder[i](e_out)
            e_result.append(e_out)
        
        res_out = self.bottleneck(e_result[-1])
        d_out = res_out.clone()
        for i in range(self.n_down):
            d_out = self.decoder.decoder[i](d_out)
            skip = e_result[self.n_down - i - 1]
            d_out = torch.cat([skip, d_out], dim=1)
            # d_out = d_out.double()
            d_out = self.decoder.skip[i](d_out)

        return self.decoder.decoder[-1](d_out)