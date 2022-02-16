from typing import List, Tuple

import torch
import torch.nn.functional as F
from scipy import io
from torch import nn

from mmderain.models.common import make_layer
from mmderain.models.registry import BACKBONES


class BasicResBlock(nn.Module):

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes)
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        out = self.act(out + x)
        return out


class MNet(nn.Module):

    def __init__(self, planes: int, num_blocks: int, tau: float = 0.5) -> None:
        super().__init__()

        self.model = make_layer(BasicResBlock, num_blocks, planes=planes)
        self.act = nn.ReLU()

        self.tau = nn.Parameter(torch.Tensor([tau]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        out = self.act(out - self.tau.expand_as(out))
        return out


class BNet(nn.Module):

    def __init__(self, bg_planes: int, feat_planes: int, num_blocks: int) -> None:
        super().__init__()

        self.planes = [bg_planes, feat_planes]
        self.model = make_layer(BasicResBlock, num_blocks, planes=bg_planes+feat_planes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        out = self.model(x)
        return torch.split(out, self.planes, dim=1)


@BACKBONES.register_module()
class RCDNet(nn.Module):
    """RCDNet Network Structure

    Paper: A Model-Driven Deep Neural Network for Single Image Rain Removal
    Official Code: https://github.com/hongwang01/RCDNet

    Args:
        mid_channels (int): Channel number of intermediate features. Default: 32
        num_stages (int): Number of intermediate stgaes. Default 17
        num_blocks (int): Block number in the trunk network. Default 4
        init_etaM (float): Initial value for \\eta_{1}. Default 1.0
        init_etaB (float): Initial value for \\eta_{2}. Default 5.0
    """

    def __init__(
        self,
        mid_channels: int = 32,
        num_stages: int = 17,
        num_blocks: int = 4,
        init_etaM: float = 1.0,
        init_etaB: float = 5.0
    ) -> None:
        super().__init__()

        in_channels = 3  # support 3-dim input only

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_stages = num_stages

        # Rain kernel
        rain_kernel = io.loadmat('mmderain/models/backbones/rcdnet_init_kernel.mat')['C9']
        rain_kernel = torch.FloatTensor(rain_kernel)

        self.C0 = nn.Parameter(rain_kernel)
        self.C = nn.Parameter(rain_kernel)

        # filter for initializing B and Z
        self.Cz = torch.empty(3, 3).fill_(1./9)
        self.Cz = nn.Parameter(self.Cz)

        # step size
        self.eta_M = nn.Parameter(torch.Tensor([init_etaM]))
        self.eta_B = nn.Parameter(torch.Tensor([init_etaB]))

        # for sparse rain layer
        self.tau = nn.Parameter(torch.Tensor([1]))

        # proximal operator
        self.prox_B = nn.ModuleList([BNet(in_channels, mid_channels, num_blocks)
                                     for _ in range(num_stages)])
        self.prox_M = nn.ModuleList([MNet(mid_channels, num_blocks) for _ in range(num_stages)])

        self.B0 = BNet(in_channels, mid_channels, num_blocks)
        self.last = BNet(in_channels, mid_channels, num_blocks)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor]]:
        listB = []
        listR = []

        # initialize B0 and Z0
        Cz = self.Cz.expand(self.mid_channels, self.in_channels, -1, -1)
        Z00 = F.conv2d(x, Cz, stride=1, padding=1)  # dual variable z
        B0, Z0 = self.B0(torch.cat([x, Z00], dim=1))

        listB.append(B0)

        # iteration variables
        B, Z, M, R = B0, Z0, None, None

        for i in range(self.num_stages):
            # M-Net updating
            R_hat = x - B
            if i == 0:
                rain_residual = F.relu(R_hat - self.tau)  # \tilde{R} - \hat{R}
                epsilon = F.conv_transpose2d(rain_residual, self.C0 / 10, stride=1, padding=4)
                M = self.prox_M[i](epsilon)
            else:
                rain_residual = R - R_hat
                epsilon = F.conv_transpose2d(rain_residual, self.C / 10,
                                             stride=1, padding=4) * self.eta_M/10
                M = self.prox_M[i](M - epsilon)

            # B-Net updating
            R = F.conv2d(M, self.C / 10, stride=1, padding=4)
            B_hat = x - R
            B_mid = (1 - self.eta_B / 10) * B + self.eta_B / 10 * B_hat
            B, Z = self.prox_B[i](torch.cat([B_mid, Z], dim=1))

            listR.append(R)
            listB.append(B)

        B, _ = self.last(torch.cat([B, Z], dim=1))
        listB.append(B)
        return listB, listR
