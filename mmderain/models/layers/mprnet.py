from functools import partial
from typing import Tuple

import torch
from torch import nn


class SAM(nn.Module):
    """Supervised Attention Module

    Paper: Multi-Stage Progressive Image Restoration.

    Args:
        planes (int): Channel number of inputs.
        out_planes (int): Channel number of outputs.
        kernel_size (int): Same as ``kernel_size`` argument in ``nn.Conv2d``.
            Default: 1.
        bias (bool): Same as ``bias`` argument in ``nn.Conv2d``.
            Default: False.
    """

    def __init__(self,
                 planes: int,
                 out_planes: int,
                 kernel_size: int = 1,
                 bias: bool = False) -> None:
        super().__init__()

        buildConv = partial(nn.Conv2d, kernel_size=kernel_size, stride=1,
                            padding=kernel_size//2, bias=bias)
        self.conv1 = buildConv(planes, planes)
        self.conv2 = buildConv(planes, out_planes)
        self.conv3 = buildConv(out_planes, planes)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor]:
        x1 = self.conv1(x)
        out = self.conv2(x) + y
        x2 = torch.sigmoid(self.conv3(y))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, out
