from typing import Optional, Tuple

import torch
from mmcv.runner import load_checkpoint
from mmderain.models.layers import SELayer
from mmderain.models.registry import BACKBONES
from mmderain.utils import get_root_logger
from torch import nn


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

    Paper: Detail-recovery Image Deraining via Context Aggregation Networks
    Official Code: https://github.com/Dengsgithub/DRD-Net
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
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f"But received {type(pretrained)}.")
