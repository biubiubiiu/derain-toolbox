from typing import Tuple

import torch
from torch import nn

from mmderain.models.layers import SELayer
from mmderain.models.registry import BACKBONES


class DCCL(nn.Module):
    """Dilated Conv Concatenation Layer"""

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=3, padding=3)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=5, padding=5)

        self.fusion = nn.Conv2d(3*planes, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, planes: int, use_se: bool = False) -> None:
        super().__init__()

        models = [
            DCCL(planes),
            nn.BatchNorm2d(planes, eps=0.01, momentum=0.99),
            nn.PReLU(init=0.),
            DCCL(planes),
            nn.BatchNorm2d(planes, eps=0.01, momentum=0.99),
        ]
        if use_se:
            models.append(SELayer(planes, reduction=planes//4))

        self.model = nn.Sequential(*models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class SubNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        num_blocks: int,
        use_se: bool
    ) -> None:
        super().__init__()

        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(init=0.)
        )

        trunk = []
        for _ in range(num_blocks):
            trunk.append(BasicBlock(mid_channels, use_se=use_se))

        trunk.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1))
        trunk.append(nn.BatchNorm2d(mid_channels, eps=0.01, momentum=0.99))
        self.trunk = nn.Sequential(*trunk)

        self.last = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feat(x)
        out = self.trunk(feat)
        out = out + feat
        out = self.last(out)
        return out


@BACKBONES.register_module()
class DRDNet(nn.Module):
    """DRD-Net Network Strcuture

    Paper: Detail-recovery Image Deraining via Context Aggregation Networks.
    Official Code: https://github.com/Dengsgithub/DRD-Net

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 64.
        num_blocks (int): Depth of each subnet. Default: 16.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 64,
        num_blocks: int = 16
    ) -> None:
        super().__init__()

        self.rain_extract = SubNet(in_channels, out_channels, mid_channels, num_blocks, use_se=True)
        self.bg_recovery = SubNet(in_channels, out_channels, mid_channels, num_blocks, use_se=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        rain = self.rain_extract(x)
        out1 = x - rain

        # bg = self.bg_recovery(x + out1)
        bg = self.bg_recovery(x)
        out2 = x - rain + bg

        return out1, out2
