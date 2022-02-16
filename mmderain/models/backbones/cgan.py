import torch
from torch import nn

from mmderain.models.common import make_layer
from mmderain.models.registry import BACKBONES, COMPONENTS


class BasicGeneratorBlock(nn.Module):

    def __init__(self, in_planes: int, out_planes: int, act: str = 'leakyrelu') -> None:
        super().__init__()

        models = []
        if act == 'leakyrelu':
            models.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == 'relu':
            models.append(nn.ReLU(inplace=True))

        models.extend([
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
        ])

        self.model = nn.Sequential(*models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@BACKBONES.register_module()
class IDGenerator(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 64
    ) -> None:
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.model0 = nn.ModuleList([BasicGeneratorBlock(mid_channels, mid_channels)
                                     for _ in range(3)])
        self.model1 = nn.Sequential(
            BasicGeneratorBlock(mid_channels, mid_channels//2),
            BasicGeneratorBlock(mid_channels//2, 1),
            BasicGeneratorBlock(1, mid_channels//2),
            BasicGeneratorBlock(mid_channels//2, mid_channels, act='relu'),
        )
        self.model2 = make_layer(BasicGeneratorBlock, 2, in_planes=mid_channels,
                                 out_planes=mid_channels, act='relu')
        self.model3 = BasicGeneratorBlock(mid_channels, mid_channels, act='relu')
        self.last = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv0(x)

        features = []
        for model in self.model0:
            out = model(out)
            features.append(out)

        feat1, _, feat3 = features

        out = self.model1(out)
        out = out + feat3
        out = self.model2(out)
        out = out + feat1
        out = self.model3(out)
        out = self.last(out)
        return out


class BasicDiscriminatorBlock(nn.Module):

    def __init__(self, in_planes: int, out_planes: int, stride: int = 2) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@COMPONENTS.register_module()
class IDDiscriminator(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 48,
        n_layer: int = 3
    ) -> None:
        super().__init__()

        models = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(n_layer-1):
            mult = min(2**(i+1), 8)
            models.append(
                BasicDiscriminatorBlock(
                    mid_channels * (mult//2),
                    mid_channels * mult,
                    stride=2
                ))

        models.append(BasicDiscriminatorBlock(mid_channels * mult, mid_channels * mult, stride=1))
        models.extend([
            nn.Conv2d(mid_channels * mult, out_channels, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        ])

        self.model = nn.Sequential(*models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
