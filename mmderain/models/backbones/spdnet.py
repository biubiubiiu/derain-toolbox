from functools import partial
from typing import List, Sequence, Tuple

import einops
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from torch import nn

from mmderain.models.common import get_rcp, make_layer, sizeof
from mmderain.models.layers import SELayer
from mmderain.models.registry import BACKBONES


class ConvAct(nn.Module):
    """2D Convolution + Activation"""

    def __init__(self, in_planes: int, out_planes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SEResBlock(nn.Module):
    """SE-ResBlock"""

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SELayer(planes, reduction=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class SRiR(nn.Module):
    """SE-ResBlock in Residual Block"""

    def __init__(self, planes: int, n_resblock: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            *[SEResBlock(planes) for _ in range(n_resblock)],
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.model(x)
        out = self.act(out)
        return out


class RCPEM(nn.Module):
    """RCP Extration Module"""

    def __init__(self, in_planes: int, out_planes: int, n_resblock: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            ConvAct(in_planes, out_planes//2),
            ConvAct(out_planes//2, out_planes),
            SRiR(out_planes, n_resblock)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = get_rcp(x).repeat(1, x.size(1), 1, 1)
        return self.model(feat)


class IFM(nn.Module):
    """Interactive Fusion Module"""

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.conv0 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(planes*2, 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(planes*2, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feat_x0 = self.conv0(x)
        feat_y0 = self.conv1(y)
        w0 = torch.sigmoid(feat_x0 * feat_y0)

        x_prime = x * w0
        y_prime = y * w0

        wx1, wx2 = torch.chunk(self.conv2(torch.cat([x, x_prime], dim=1)), chunks=2, dim=1)
        wy1, wy2 = torch.chunk(self.conv3(torch.cat([y, y_prime], dim=1)), chunks=2, dim=1)

        out_x = x*wx1 + x_prime*wx2
        out_y = y*wy1 + y_prime*wy2

        out = torch.cat([out_x, out_y], dim=1)
        return out


class WMLMDecomposition(nn.Module):

    def __init__(self, planes: int, is_first_level: bool) -> None:
        super().__init__()

        self.is_first_level = is_first_level

        self.dwt = DWTForward(J=1, wave='haar')
        self.conv = ConvAct(planes*2, planes) if is_first_level else ConvAct(planes*4, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first_level:
            return self.conv(x)
        else:
            return self.conv(self._decomp(x))

    def _decomp(self, x: torch.Tensor) -> torch.Tensor:
        xl, xh = self.dwt(x)
        xl = xl.unsqueeze(2)
        feat = torch.cat([xh[0], xl], dim=2)
        out = einops.rearrange(feat, 'b c n h w -> b (n c) h w')
        return out


class WMLMFusion(nn.Module):

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.idwt = DWTInverse(wave='haar')

        self.conv = ConvAct(planes, planes*4)
        self.upsample = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=2),
            nn.ReLU(inplace=True)
        )
        self.last = nn.Sequential(
            SEResBlock(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self._reconstruct(x)
        x = self.upsample(x)
        x = F.interpolate(x, size=sizeof(y))
        y = x + y
        return self.last(y)

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        feat = einops.rearrange(x, 'b (c1 c2) h w -> b c1 c2 h w', c2=4)
        xl, xh = torch.split(feat, [1, 3], dim=2)
        xl = xl.squeeze(dim=2)
        out = self.idwt((xl, [xh]))
        return out


class WMLM(nn.Module):
    """Wavelet-based Multi-level Module"""

    def __init__(self, planes: int, n_level: int, n_srir: int, n_resblock: int) -> None:
        super().__init__()

        self.decomposition = nn.ModuleList([
            WMLMDecomposition(planes, is_first_level=(i == 0))
            for i in range(n_level)
        ])

        self.trunks = nn.ModuleList([
            make_layer(SRiR, n_srir, planes=planes, n_resblock=n_resblock)
            for _ in range(n_level)
        ])

        self.fusions = nn.ModuleList([
            WMLMFusion(planes)
            for _ in range(n_level-1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_features = []

        prev = x
        for model in self.decomposition:  # level 0, level 1, ... (top-down)
            prev = model(prev)
            init_features.append(prev)

        out_features = []
        for init_feat, model in zip(init_features, self.trunks):
            feat = model(init_feat)
            out_features.append(feat)

        out = out_features.pop()  # feature from bottom level
        for model in self.fusions:
            out = model(out, out_features.pop())  # bottom-up fusion

        return out


class Subnet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        n_level: int,
        n_srir: int,
        n_resblock: int,
        index: int
    ) -> None:

        super().__init__()

        if index > 0:
            conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)
            self.fusion1 = conv3x3(mid_channels*(index+1), mid_channels)
            self.fusion2 = conv3x3(mid_channels*(index+1), mid_channels)
        else:
            self.fusion1 = nn.Identity()
            self.fusion2 = nn.Identity()

        self.rcpem = RCPEM(in_channels, mid_channels, n_resblock)
        self.ifm = IFM(mid_channels)
        self.wmlm = WMLM(mid_channels, n_level, n_srir, n_resblock)
        self.last = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor]:
        rcp_feat = self.rcpem(x)
        feat0 = self.fusion1(torch.cat(feats, dim=1))
        feat1 = self.ifm(feat0, rcp_feat)
        feat2 = self.wmlm(feat1)
        feat3 = self.fusion2(torch.cat([feat2] + feats[:-1], dim=1))
        out = self.last(feat3)
        return out, feat2


@BACKBONES.register_module()
class SPDNet(nn.Module):
    """SPDNet Network Structure

    Paper: Structure-Preserving Deraining with Residue Channel Prior Guidance
    Official Code: https://github.com/Joyies/SPDNet

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 32.
        n_stage (int): Number of stages. Default: 3.
        n_level (int): Number of levels in WMLM. Default: 3.
        n_srir (int): Number of SRiR blocks of each level in WMLM. Default: 3.
        n_resblock (int): Number of Resblocks in SRiR Module. Default: 3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 32,
        n_stage: int = 3,
        n_level: int = 3,
        n_srir: int = 3,
        n_resblock: int = 3
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            ConvAct(in_channels, mid_channels//2),
            ConvAct(mid_channels//2, mid_channels),
        )

        self.subnets = nn.ModuleList([
            Subnet(in_channels, out_channels, mid_channels, n_level, n_srir, n_resblock, i)
            for i in range(n_stage)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        prev_out = x

        init_feat = self.head(x)
        features = [init_feat]

        for net in self.subnets:
            out, feat = net(prev_out, features)
            prev_out = out
            outputs.append(out)
            features.insert(0, feat)

        return outputs
