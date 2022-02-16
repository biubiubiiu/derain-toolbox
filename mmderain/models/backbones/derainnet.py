import torch
from torch import nn

from mmderain.models.common import GuidedFilter2d
from mmderain.models.registry import BACKBONES


@BACKBONES.register_module()
class DerainNet(nn.Module):
    """DerainNet Network Structure

    Paper: Clearing the Skies: A Deep Network Architecture for Single-Image Rain Removal
    Official Code: https://xueyangfu.github.io/projects/tip2017.html

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 512
        padding (int): Padding added to all four sides of the input. Default: 10
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 512,
        padding: int = 10,
    ) -> None:

        super().__init__()

        self.guided_filter = GuidedFilter2d(radius=15, eps=1)

        self.padding = padding
        self.pad = nn.ReflectionPad2d(padding)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=8, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)

        # extracts high frequency component
        lf = self.guided_filter(x, x)
        out = x - lf

        out = self.model(out)

        # crop to original shape
        lf = lf[:, :, self.padding:-self.padding, self.padding:-self.padding]
        out = out[:, :, self.padding-4:-self.padding+4, self.padding-4:-self.padding+4]

        return out + lf
