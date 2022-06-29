import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn.utils import weight_norm

from layer.conv import LeakyReLUConv2d, Conv2dBlock
from layer.norm import spectral_norm, AdaptiveInstanceNorm2d

####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class DisContent(nn.Module):
    def __init__(self):
        super(DisContent, self).__init__()
        model = []
        model += [LeakyReLUConv2d(256, 256, kernel_size=3, stride=2, padding=1, norm='None')] # Instance
        model += [LeakyReLUConv2d(256, 256, kernel_size=3, stride=2, padding=1, norm='None')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=3, stride=2, padding=1, norm='None')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=3, stride=2, padding=1, norm='None')]
        # model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        return out


class MSDiscriminator(nn.Module):
    def __init__(self, input_dim, dim, n_layer, norm, activation, n_scales, pad_type, **kwargs):
        super(MSDiscriminator, self).__init__()
        self.n_layer = n_layer
        self.dim = dim
        self.norm = norm
        self.activation = activation
        self.num_scales = n_scales
        self.pad_type = pad_type
        self.input_dim = input_dim

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()

        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):

        dim = self.dim

        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm=None, activation=self.activation, pad_type=self.pad_type)]

        for _ in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activation, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs
