from typing import List, Optional, Tuple

import torch
from torch import nn

from mmderain.models.layers import RESCAN_GRU, SELayer
from mmderain.models.registry import BACKBONES


class ResidualBlock(nn.Module):

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=3e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=3e-4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class REU(nn.Module):
    """Recurrent Enhancement Unit"""

    def __init__(self, in_planes: int, planes: int) -> None:
        super().__init__()

        self.rnn = RESCAN_GRU(in_planes, planes, kernel_size=3, dilation=1)
        self.last = nn.Sequential(
            SELayer(planes, reduction=4),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        state = self.rnn(x, hx)
        out = self.last(state)
        return out, out


class HEU(nn.Module):
    """Hierachy Enhancement Unit"""

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.res0 = ResidualBlock(planes)
        self.res1 = ResidualBlock(planes)
        self.last = nn.Sequential(
            nn.Conv2d(planes*3, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=3e-4),
            SELayer(planes, reduction=4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.res0(x)
        out2 = self.res1(out1)
        out = torch.cat([x, out1, out2], dim=1)
        out = self.last(out)
        out = out + x
        return out


class ReHEB(nn.Module):
    """Recurrent Hierachy Enhancement Block"""

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.heu = HEU(planes)
        self.reu = REU(planes, planes)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        out = self.heu(x)
        out, mem = self.reu(out, hx)
        return out, mem


class RMG(nn.Module):
    """Residual Map Generator"""

    def __init__(self, planes: int, out_planes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            SELayer(planes, reduction=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes, out_planes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@BACKBONES.register_module()
class ReHEN(nn.Module):
    """ReHEN Network Structure

    Paper: Single Image Deraining via Recurrent Hierarchy Enhancement Network
    Official Code: https://github.com/nnUyi/ReHEN

    Args:
        in_out_channels (int): Channel number of inputs and outputs.
        mid_channels (int): Channel number of intermediate features. Default: 24
        n_stages (int): Number of stages. Default: 4
        n_blocks (int): Number of ReHEBs in network. Default: 5
    """

    def __init__(
        self,
        in_out_channels: int,
        mid_channels: int = 24,
        n_stages: int = 4,
        n_blocks: int = 5
    ) -> None:
        super().__init__()

        self.n_stages = n_stages
        self.n_blocks = n_blocks

        self.models = nn.ModuleList([
            REU(in_out_channels, mid_channels) if i == 0 else ReHEB(mid_channels)
            for i in range(n_blocks)
        ])
        self.last = RMG(mid_channels, in_out_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        memories = [None] * self.n_blocks
        outputs = []

        out = x
        for _ in range(self.n_stages):
            for i, (mem, model) in enumerate(zip(memories, self.models)):
                out, state = model(out, mem)
                memories[i] = state

            res = self.last(out)  # rain residual
            out = x - res
            outputs.append(out)

        return outputs
