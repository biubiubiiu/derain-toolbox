from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mmderain.models.common import sizeof
from mmderain.models.registry import BACKBONES
from mmderain.models.layers import SELayer_Modified


class ResidualBlock(nn.Module):

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=planes)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        out = self.act(out + x)
        return out


class FFRB(nn.Module):
    """Feature fusion residual block"""

    def __init__(
        self,
        in_planes: int,
        mid_planes: int,
        out_planes: int,
        kernel_size: int
    ) -> None:
        super().__init__()

        inter_planes = mid_planes * 4
        planes_per_group = 4

        self.model0 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=inter_planes // planes_per_group, num_channels=inter_planes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inter_planes, mid_planes,
                      kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.GroupNorm(num_groups=mid_planes//planes_per_group, num_channels=mid_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model1 = nn.Sequential(
            nn.Conv2d(in_planes+mid_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            SELayer_Modified(out_planes, reduction=out_planes//6,
                             bias=True, act=nn.LeakyReLU(0.2, inplace=True))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model0(x)
        out = torch.cat([x, out], dim=1)
        out = self.model1(out)
        return out


class Encoder(nn.Module):

    def __init__(self, layers: List[nn.Module]) -> None:
        super().__init__()

        self.models = nn.ModuleList(layers)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor]]:
        features = []
        out = x
        for model in self.models:
            out = model(out)
            features.append(out)
            out = self.downsample(out)

        return out, features


class Decoder(nn.Module):

    def __init__(self, layers: List[nn.Module]) -> None:
        super().__init__()

        self.models = nn.ModuleList(layers)
        self.upsample = partial(F.interpolate, mode='nearest', align_corners=None)

    def forward(self, x: torch.Tensor, bridges: Sequence[torch.Tensor]) -> torch.Tensor:
        features = []
        out = x
        for model in self.models:
            out = model(out)
            out = self.upsample(out, scale_factor=2)
            out = torch.cat([out, bridges.pop()], dim=1)
            features.append(out)
        return out, features


class UFFRB(nn.Module):
    """U-Net structure constructed with FFRBs"""

    def __init__(self, planes: int, depth: int) -> None:
        super().__init__()

        ffrb_builder = partial(FFRB, mid_planes=planes, out_planes=planes, kernel_size=3)

        self.encoder = Encoder([ffrb_builder(in_planes=planes) for _ in range(depth // 2)])
        self.decoder = Decoder([ffrb_builder(in_planes=planes if i == 0 else planes*2)
                                for i in range(depth//2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out, encoder_features = self.encoder(out)
        out, _ = self.decoder(out, encoder_features)
        return out


class Foundation(nn.Module):
    """Structure for feature compensator and error detector"""

    def __init__(
        self,
        in_planes: int,
        mid_planes: int,
        out_planes: int,
        uffrb_depth: int = -1,
        n_ffrb: int = 3,
        act: Optional[str] = None
    ) -> None:
        super().__init__()

        models = []

        planes_per_group = 4
        models.extend([
            nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=mid_planes//planes_per_group, num_channels=mid_planes),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        use_uffrb = uffrb_depth > 0
        if use_uffrb:
            models.append(UFFRB(mid_planes, uffrb_depth))
        for i in range(n_ffrb):
            if use_uffrb and i == 0:
                models.append(FFRB(mid_planes*2, mid_planes, mid_planes, kernel_size=3))
            else:
                models.append(FFRB(mid_planes, mid_planes, mid_planes, kernel_size=3))

        models.append(nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1))
        if act == 'leakyrelu':
            models.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == 'sigmoid':
            models.append(nn.Sigmoid())

        self.model = nn.Sequential(*models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FeatureCompensator(Foundation):

    def __init__(
        self,
        in_planes: int,
        mid_planes: int,
        out_planes: int,
        use_uffrb: bool = True,
        n_ffrb: int = 3,
        act: str = 'leakyrelu'
    ) -> None:
        uffrb_depth = 6 if use_uffrb else -1
        super().__init__(in_planes, mid_planes, out_planes, uffrb_depth, n_ffrb, act)


class ErrorDetector(Foundation):

    def __init__(
        self,
        in_planes: int,
        mid_planes: int,
        out_planes: int,
        use_uffrb: bool,
        n_ffrb: int,
        act: Optional[str] = None
    ) -> None:
        uffrb_depth = 6 if use_uffrb else -1
        super().__init__(in_planes, mid_planes, out_planes, uffrb_depth, n_ffrb, act)


class Refinement(nn.Module):
    """Refinement Module"""

    def __init__(
        self,
        in_planes: int,
        mid_planes: int,
        out_planes: int,
        n_scale: int,
        n_residual: int
    ) -> None:
        super().__init__()

        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True)
            )
            for _ in range(n_scale)
        ])
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes+4, mid_planes, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stacked_rb = nn.ModuleList([
            ResidualBlock(mid_planes)
            for _ in range(n_residual)
        ])
        self.use_feature_idxs = [0, 3, 6]
        self.last = nn.Sequential(
            nn.Conv2d(mid_planes * len(self.use_feature_idxs), mid_planes,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_planes, mid_planes // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_planes//2, out_planes, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample = partial(F.interpolate, mode='nearest', align_corners=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mutli-scale processing
        out_shape = sizeof(x)
        outputs = []
        for i, model in enumerate(self.multi_scale):
            tmp = F.avg_pool2d(x, kernel_size=2**(i+2))
            tmp = model(tmp)
            tmp = self.upsample(tmp, size=out_shape)
            outputs.append(tmp)
        multi_scale_out = torch.cat(outputs, dim=1)

        # pass through stacked residual blocks
        out = torch.cat([multi_scale_out, x], dim=1)
        out = self.conv0(out)
        features = []
        for i, model in enumerate(self.stacked_rb):
            out = model(out)
            if i in self.use_feature_idxs:
                features.append(out)
        out = torch.cat(features, dim=1)
        out = self.last(out)
        return out


@BACKBONES.register_module()
class RLNet(nn.Module):
    """DerainRLNet Network Structure

    Paper: Robust Representation Learning with Feedback for Single Image Deraining
    Official Code: https://github.com/LI-Hao-SJTU/DerainRLNet

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (list[int]): Channel number of intermediate features. Default: [24, 32, 18]
        theta (list[float]): Values of theta1 and theta2. Default: [0.15, 0.05]
        n_scale (int): Number of scales in refinement module. Default: 4
        n_residual (int): Number of residual blocks in refinement module. Default: 7
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: List[int] = [24, 32, 18],
        theta: List[float] = [0.15, 0.05],
        n_scale: int = 4,
        n_residual: int = 7
    ) -> None:
        super().__init__()

        theta1, theta2 = theta
        self.theta1 = theta1
        self.theta2 = theta2

        mid0, mid1, mid2 = mid_channels

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, mid0, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # separate branches
        encoder_builder = partial(FFRB, mid0, mid0, mid0)

        self.encoder1 = Encoder([encoder_builder(kernel_size=3) for _ in range(3)])
        self.encoder2 = Encoder([encoder_builder(kernel_size=5) for _ in range(3)])
        self.encoder3 = Encoder([encoder_builder(kernel_size=7) for _ in range(3)])

        decoder_builders = [
            partial(FFRB, mid0, mid0, mid0),
            partial(FFRB, mid0*2+3*out_channels, mid0, mid0),
            partial(FFRB, mid0*2+3*out_channels, mid0, mid0)
        ]

        self.decoder1 = Decoder([f(kernel_size=3) for f in decoder_builders])
        self.decoder2 = Decoder([f(kernel_size=5) for f in decoder_builders])
        self.decoder3 = Decoder([f(kernel_size=7) for f in decoder_builders])

        # feature compensators
        self.fc1_internal = FeatureCompensator(3*mid0, mid1, out_channels)
        self.fc2_internal = FeatureCompensator(3*mid0, mid1, out_channels)

        self.fc1_externel = FeatureCompensator(in_channels, mid1, out_channels,
                                               use_uffrb=False, n_ffrb=1, act='sigmoid')
        self.fc2_externel = FeatureCompensator(in_channels, mid1, out_channels,
                                               use_uffrb=False, n_ffrb=1, act='sigmoid')

        # error detectors
        self.ed1 = ErrorDetector(3*(mid0*2+3*out_channels), mid1, out_channels,
                                 use_uffrb=False, n_ffrb=5, act='leakyrelu')
        self.ed2 = ErrorDetector(in_channels+out_channels, mid1, out_channels,
                                 use_uffrb=True, n_ffrb=4, act='sigmoid')

        # post processor
        self.fusion = nn.Sequential(
            nn.Conv2d(6*mid0, mid2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.refinement = Refinement(mid2 + 6*out_channels, mid1, out_channels, n_scale, n_residual)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = partial(F.interpolate, mode='nearest', align_corners=None)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        out = self.conv0(x)

        out1, features1 = self.encoder1(out)
        out2, features2 = self.encoder2(out)
        out3, features3 = self.encoder3(out)

        # feature compensation
        FA1, FA2, FA3 = features1[1], features2[1], features3[1]
        FB1, FB2, FB3 = features1[2], features2[2], features3[2]

        F1 = self.fc1_internal(torch.cat([FA1, FA2, FA3], dim=1))
        F2 = self.fc2_internal(torch.cat([FB1, FB2, FB3], dim=1))

        refined1 = [features1[0]] + \
            [torch.cat([FA1, F1, F1, F1], dim=1), torch.cat([FB1, F2, F2, F2], dim=1)]
        refined2 = [features2[0]] + \
            [torch.cat([FA2, F1, F1, F1], dim=1), torch.cat([FB2, F2, F2, F2], dim=1)]
        refined3 = [features3[0]] + \
            [torch.cat([FA3, F1, F1, F1], dim=1), torch.cat([FB3, F2, F2, F2], dim=1)]

        out1, dec_feat1 = self.decoder1(out1, refined1)
        out2, dec_feat2 = self.decoder2(out2, refined2)
        out3, dec_feat3 = self.decoder3(out3, refined3)

        # error detection
        FE1, FE2, FE3 = dec_feat1[1], dec_feat2[1], dec_feat3[1]

        phi1 = self.ed1(torch.cat([FE1, FE2, FE3], dim=1))
        phi = self.ed2(torch.cat([self.pool(x), phi1], dim=1))
        err = torch.div(self.theta1, phi) - self.theta1
        phi1_prime = F.relu(phi1-err*(1-2*phi1), inplace=True)
        phi1_prime = self.upsample(phi1_prime, scale_factor=2)

        # post processing
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(out)

        # inject error information
        out = torch.cat([out, phi1_prime, phi1_prime, phi1_prime,
                        phi1_prime, phi1_prime, phi1_prime], dim=1)

        # refine
        out = self.refinement(out)

        if y is None:
            return out, F1, F2, phi1, phi
        else:
            y2 = F.avg_pool2d(y, kernel_size=2, stride=2)
            y4 = F.avg_pool2d(y, kernel_size=4, stride=4)
            k2 = self.fc1_externel(y2)
            k4 = self.fc2_externel(y4)
            y2 = y2 + self.theta2 * self.theta2 * k2 * y2
            y4 = y4 + self.theta2 * self.theta2 * k4 * y4
            return out, F1, F2, phi1, phi, y2, y4, k2, k4
