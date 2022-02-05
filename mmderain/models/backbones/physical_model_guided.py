from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmderain.models.common import make_layer
from mmderain.models.registry import BACKBONES
from mmderain.utils import get_root_logger
from torch import nn


def sizeof(x: torch.Tensor) -> Tuple[int]:
    return tuple(x.shape)[2:]


class MSRB(nn.Module):
    """Multi-Scale Residual Block"""

    def __init__(self, planes: int, scales: List[int]) -> None:
        super().__init__()

        self.downsamplings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=s, stride=s)
            for s in scales
        ])
        self.upsample = partial(F.interpolate, mode='bilinear', align_corners=False)

        self.model = nn.Sequential(
            nn.Conv2d(len(scales)*planes, planes, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [
            self.upsample(input=downsample(x), size=sizeof(x))
            for downsample in self.downsamplings
        ]
        out = torch.cat(feats, dim=1)
        out = self.model(out)
        return out + x


class MSDC(nn.Module):

    def __init__(self, in_planes: int, planes: int) -> None:
        super(MSDC, self).__init__()

        # NOTE: The original code use convolutions with different kernel sizes.
        # Following the settings in the paper, use convolutions with different
        # dilation rates here
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1,  padding=3, dilation=3)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.fusion = nn.Conv2d(3 * planes, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.act(self.conv1(x))
        conv2 = self.act(self.conv2(x))
        conv3 = self.act(self.conv3(x))
        out = self.fusion(torch.cat([conv1, conv2, conv3], dim=1))
        return out


class Stem(nn.Module):

    def __init__(self, planes: int, scales: List[int], depth: int) -> None:
        super().__init__()

        assert depth % 2 == 1, 'depth should be odd'

        self.encoder = nn.ModuleList([
            make_layer(MSRB, 2, planes=planes, scales=scales)
            for _ in range(depth // 2)
        ])
        self.bottleneck = make_layer(MSRB, 2, planes=planes, scales=scales)
        self.decoder = nn.ModuleList([
            make_layer(MSRB, 2, planes=planes, scales=scales)
            for _ in range(depth // 2)
        ])

        self.bridges = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2*planes, planes, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True)
            )
            for _ in range(depth // 2)
        ])

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = partial(F.interpolate, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        encoder_features = []
        for i, model in enumerate(self.encoder):
            out = model(out) if i == 0 else model(self.downsample(out))
            encoder_features.append(out)

        out = self.bottleneck(self.downsample(out))

        for model, skip in zip(self.decoder, self.bridges):
            bridge = encoder_features.pop()
            out = model(skip(torch.cat([
                self.upsample(input=out, size=sizeof(bridge)),
                bridge
            ], dim=1)))

        return out


class Subnet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        depth: int,
        scales: List[int],
        use_msdc: bool,
    ) -> nn.Module:
        super().__init__()

        models = []
        if use_msdc:
            models.append(MSDC(in_channels, mid_channels))
        else:
            models.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        models.append(Stem(planes=mid_channels, scales=scales, depth=depth))
        models.append(nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        ))
        self.model = nn.Sequential(*models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@BACKBONES.register_module()
class PhysicalModelGuided(nn.Module):
    """Physical Model Guided Image Deraining Network Structure

    Paper: Physical Model Guided Deep Image Deraining
    Official Code: https://github.com/Ohraincu/PHYSICAL-MODEL-GUIDED-DEEP-IMAGE-DERAINING

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 64
        depth (int): Depth of each subnet. Default: 5
        scales (list[int]): Scales in multi-scale residual block. Default: [1, 2, 4]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 64,
        depth: int = 5,
        scales: List[int] = [1, 2, 4]
    ) -> None:
        super().__init__()

        self.rain_streak_net = Subnet(in_channels, out_channels, mid_channels,
                                      depth=depth, scales=scales, use_msdc=False)

        self.rain_free_net = Subnet(in_channels, out_channels, mid_channels,
                                    depth=depth, scales=scales, use_msdc=False)

        self.guided_learning_net = Subnet(2*out_channels, out_channels, mid_channels,
                                          depth=depth, scales=scales, use_msdc=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        R_hat = self.rain_streak_net(x)
        B_hat = self.rain_free_net(x)
        B_refine = self.guided_learning_net(torch.cat([R_hat, B_hat], dim=1))
        return R_hat, B_hat, B_refine

    def init_weights(self, pretrained: Optional[str], strict: bool = True):
        """Init weights for models

        Args:
            pretrained (str | optional): Path to the pretrained model.
            strict (bool): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # use default initialization
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f"But received {type(pretrained)}.")
