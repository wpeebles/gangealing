import torch.nn as nn

from .functional import splat2d

__all__ = ['Splat2D', 'splat2d']


class Splat2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coordinates, values, sigma, height, width):
        return splat2d(coordinates, values, sigma, height, width)

    def extra_repr(self):
        return ''
