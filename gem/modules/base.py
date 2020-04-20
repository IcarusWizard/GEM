import numpy as np

import torch
from torch.functional import F
from functools import partial

class Flatten(torch.nn.Module):
    """
        Flatten a batch of tensors
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Unflatten(torch.nn.Module):
    """ 
        Reverse the Flatten operation

        Inputs:

            shape : list, shape of the origin tensor (without batch)
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)

ACTIVATION_MAP = {
    'relu' : partial(torch.nn.ReLU, inplace=True),
    'elu' : torch.nn.ELU,
    'leakyrelu' : partial(torch.nn.LeakyReLU, negative_slope=0.1, inplace=True)
}

class MLP(torch.nn.Module):
    r"""
        Multi-layer Perceptron without activation on the output

        Inputs: 
        
            in_features : int, features numbers of the input
            out_features : int, features numbers of the output
            hidden_features : int, features numbers of the hidden layers
            hidden_layers : int, numbers of the hidden layers
            activation : str, activation function used in hidden layers, default : 'relu'
    """
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, activation='relu'):
        super(MLP, self).__init__()

        activation = ACTIVATION_MAP[activation]

        if hidden_layers == 0:
            self.net = torch.nn.Linear(in_features, out_features)
        else:
            net = []
            for i in range(hidden_layers):
                net += [
                    torch.nn.Linear(in_features if i == 0 else hidden_features, hidden_features),
                    activation()
                ]
            net.append(torch.nn.Linear(hidden_features, out_features))
            self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class ResBlock(torch.nn.Module):
    r"""
        Basic residual block 1x1 -> 3x3 -> 1x1

        Inputs: 
        
            features : int, features numbers
            batchnorm : bool, whether to use batchnorm, default : False
    """
    def __init__(self, features, batchnorm=False):
        super(ResBlock, self).__init__()
        if batchnorm:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(features, features, 1, stride=1, padding=0),
                torch.nn.BatchNorm2d(features),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(features, features, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(features),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(features, features, 1, stride=1, padding=0)
            )            
        else:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(features, features, 1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(features, features, 3, stride=1, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(features, features, 1, stride=1, padding=0)
            )

    def forward(self, x):
        return x + self.shortcut(x)

class ResNet(torch.nn.Module):
    r"""
        A simple ResNet that stack multiple residual blocks

        Inputs:

            in_channels : int, channels of the input feature map
            out_channels : int, channels of the output feature map
            featues : int, number of features used in residual blocks
            number_blocks : int, number of the stacked blocks
            batchnorm : bool, whether to use batchnorm, default : False
            zero_init : bool, whether to initialize the output as 0, default: False
    """
    def __init__(self, in_channels, out_channels, features=256, number_blocks=8, batchnorm=False, zero_init=False):
        super(ResNet, self).__init__()
        self.in_conv = torch.nn.Conv2d(in_channels, features, 3, stride=1, padding=1)
        self.res_blocks = torch.nn.ModuleList([ResBlock(features, batchnorm) for _ in range(number_blocks)])
        self.out_conv = torch.nn.Conv2d(features, out_channels, 3, stride=1, padding=1)

        if zero_init:
            with torch.no_grad():
                self.out_conv.weight.fill_(0)
                self.out_conv.bias.fill_(0)

    def forward(self, x):
        h = self.in_conv(x)
        for block in self.res_blocks:
            h = block(h)
        return self.out_conv(F.relu(h, inplace=True))

class MaskLinear(torch.nn.Linear):
    r"""
        A masked version of Linear layer,
        only with a mask : np.ndarray[in_features, out_features]
    """
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features)
        self.register_buffer('mask', torch.from_numpy(mask).float().t())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class MaskConv(torch.nn.Conv2d):
    r"""
        A masked version of Conv2d,
        only with a mask : np.ndarray[in_features, out_features, kernel_size, kernel_size]
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.register_buffer('mask', torch.from_numpy(mask).float())

    def forward(self, input):
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)