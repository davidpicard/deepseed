import torch
from torch import nn


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
    op = [
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    ]
    if bn:
        op.append(nn.BatchNorm2d(channels_out))
    if activation:
        op.append(nn.ReLU(inplace=True))
    return nn.Sequential(*op)


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


def build_network(num_class=10):
  return nn.Sequential(
    conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2),

    conv_bn(64, 128, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2),
    Residual(nn.Sequential(
        conv_bn(128, 128),
        conv_bn(128, 128),
    )),

    conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2),

    conv_bn(256, 256, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2),
    Residual(nn.Sequential(
        conv_bn(256, 256),
        conv_bn(256, 256),
    )),

    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(256, num_class, bias=False),
    Mul(0.1)
  )