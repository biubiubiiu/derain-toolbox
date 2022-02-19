from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from mmderain.models.common import gaussian_kernel_cv2, laplacian_pyramid, pyr_upsample
from mmderain.models.registry import BACKBONES


def upsample(x, output_shape):
    kernel = 4*gaussian_kernel_cv2(x.shape[1], kernel_size=5, sigma=0).to(x)
    out = pyr_upsample(x, kernel, 1-output_shape[2] % 2, 1-output_shape[3] % 2)
    return out


class RecursiveBlock(nn.Module):

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.conv0 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv0(x))
        out = self.act(self.conv1(out))
        out = self.act(self.conv2(out))
        return out


class Subnet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        n_recursion: int
    ) -> None:
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.recursive = RecursiveBlock(mid_channels)
        self.last = nn.Sequential(nn.Conv2d(mid_channels, out_channels,
                                  kernel_size=1, stride=1, padding=0))

        self.n_recursion = n_recursion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv0(x)
        out = feat
        for _ in range(self.n_recursion):
            out = self.recursive(out) + feat
        out = self.last(out)
        out = x + out
        return out


@BACKBONES.register_module()
class LPNet(nn.Module):
    """LPNet Network Structure

    Paper: Lightweight Pyramid Networks for Image Deraining.
    Official Code: https://xueyangfu.github.io/projects/LPNet.html

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 16
        max_level (int): Number of levels in Laplacian Pyramid. Default: 5.
        n_recursion (int): Number of recursions in RecursiveBlocks. Default: 5.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 16,
        max_level: int = 5,
        n_recursion: int = 5
    ) -> None:
        super().__init__()

        self.max_level = max_level

        self.subnets = nn.ModuleList([
            Subnet(in_channels, out_channels, mid_channels//(2**i), n_recursion)
            for i in reversed(range(max_level))
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        pyramid, last = laplacian_pyramid(x, kernel_size=5, sigma=0, n_levels=self.max_level-1,
                                          gauss_coeff_backend='cv2', keep_last=True)
        pyramid.insert(0, last)

        outputs = []
        prev_out = None
        for xi, subnet in zip(pyramid, self.subnets):
            out = subnet(xi)
            if prev_out is not None:
                out = out + upsample(prev_out, output_shape=out.shape)
            out = F.relu(out, inplace=True)
            outputs.append(out)
            prev_out = out

        return outputs
