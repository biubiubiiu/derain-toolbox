import torch
from torch import nn
from mmcv.runner import load_checkpoint
from typing import Optional

from mmderain.utils import get_root_logger
from mmderain.models.common import GuidedFilter2d
from mmderain.models.registry import BACKBONES


class BasicBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int, activation: bool = True) -> None:
        super().__init__()

        model = [
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes, eps=0.01, momentum=0.99),
        ]
        if activation:
            model.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class DDN(nn.Module):
    """DDN Network Structure

    Paper: Removing Rain from Single Images via a Deep Detail Network.
    Official Code: https://xueyangfu.github.io/paper/2017/cvpr/CVPR17_training_code.zip

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 16
        num_blocks (int): Block number in the trunk network. Default: 24
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 16,
        num_blocks: int = 24,
    ) -> None:

        super().__init__()

        self.guided_filter = GuidedFilter2d(radius=15, eps=1)
        self.layer1 = BasicBlock(in_channels, mid_channels)
        self.layer2 = nn.Sequential(*[
            BasicBlock(mid_channels, mid_channels)
            for _ in range(num_blocks)
        ])
        self.layer3 = BasicBlock(mid_channels, out_channels, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extracts high frequency component
        lf = self.guided_filter(x, x)
        out = x - lf

        out = self.layer1(out)
        shortcut = out

        for idx, block in enumerate(self.layer2):
            out = block(out)
            if idx % 2 > 0:  # residual conncetion every two blocks
                out = out + shortcut
                shortcut = out

        out = self.layer3(out)
        return x + out

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
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')
