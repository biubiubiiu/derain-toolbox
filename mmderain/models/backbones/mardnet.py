from functools import partial

import torch
from torch import nn

from mmderain.models.layers import SELayer_Modified
from mmderain.models.registry import BACKBONES


class ConvAct(nn.Module):
    """2D Convolution + Activation"""

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, leaky_slope=0.2) -> None:
        super().__init__()

        assert leaky_slope >= 0

        self.model = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.LeakyReLU(leaky_slope, inplace=True) if leaky_slope > 0 else nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SABlock(nn.Module):
    """Spatial Attention Block"""

    def __init__(self, planes: int, reduction: int) -> None:
        super().__init__()

        conv = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)

        self.model = nn.Sequential(
            conv(planes, planes//reduction),
            nn.ReLU(inplace=True),
            conv(planes//reduction, planes//reduction),
            nn.ReLU(inplace=True),
            conv(planes//reduction, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.model(x)
        return x*weight


class MSRB(nn.Module):
    """Multi-Scale Attention Residual Block"""

    def __init__(self, in_planes: int, out_planes: int, reduction: int):
        super().__init__()

        self.conv00 = ConvAct(in_planes, out_planes, kernel_size=3)
        self.conv10 = ConvAct(in_planes, out_planes, kernel_size=5)
        self.conv01 = ConvAct(in_planes, out_planes, kernel_size=3)
        self.conv11 = ConvAct(in_planes, out_planes, kernel_size=5)

        self.last = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            SELayer_Modified(out_planes, reduction=reduction, bias=True),
            SABlock(out_planes, reduction=reduction)
        )

    def forward(self, x):
        out0 = self.conv00(x) + self.conv10(x)
        out1 = self.conv01(out0)
        out2 = self.conv11(out0)
        out = torch.cat([out1, out2], dim=1)
        out = self.last(out)
        return out + x


@BACKBONES.register_module()
class MARDNet(nn.Module):
    """MARDNet Network Structure

    Paper: Multi-scale Attentive Residual Dense Network for Single Image Rain Removal
    Official Code: https://github.com/cxtalk/MARD-Net

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 32.
        growth_rate (int): Number of output channels in each dense layer. Default: 32.
        n_block (int): Number of MSRB blocks. Default: 8.
        att_reduction (int): Reduction ratio in spatial and channel attention blocks. Default: 4.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 32,
        growth_rate: int = 32,
        n_block: int = 8,
        att_reduction: int = 4,
    ) -> None:
        super().__init__()

        self.head = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(n_block):
            if i == 0:
                self.blocks.append(MSRB(mid_channels, growth_rate, att_reduction))
            else:
                self.blocks.append(nn.Sequential(
                    ConvAct(mid_channels+i*growth_rate, mid_channels, kernel_size=1, leaky_slope=0),
                    MSRB(mid_channels, growth_rate, att_reduction)
                ))

        self.last = nn.Sequential(
            ConvAct(mid_channels+n_block*growth_rate, mid_channels, kernel_size=1, leaky_slope=0),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        initial_feat = self.head(x)

        features = [initial_feat]
        for model in self.blocks:
            new_feature = model(torch.cat(features, dim=1))
            features.append(new_feature)

        out = torch.cat(features, dim=1)
        rain_residual = self.last(out)
        return x + rain_residual
