from typing import List, Tuple, Union

import torch
from torch import nn

from mmderain.models.common import make_layer
from mmderain.models.layers import ConvGRU, ConvLSTM
from mmderain.models.registry import BACKBONES


class ResidualBlock(nn.Module):

    def __init__(self, planes: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class ResBlockBody(nn.Module):

    def __init__(self, planes: int, num_forwards: int, recursive: bool) -> None:
        super().__init__()

        self.num_forwards = num_forwards
        self.recursive = recursive

        if recursive:
            self.model = ResidualBlock(planes)
        else:
            self.model = make_layer(ResidualBlock, num_forwards, planes=planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.recursive:
            for _ in range(self.num_forwards-1):
                out = self.model(out)
        return out


@BACKBONES.register_module()
class PRN(nn.Module):
    """PRN Network Structure.

    This is the simplified version of PReNet, where recurrent unit is removed.

    Paper: Progressive Image Deraining Networks: A Better and Simpler Baseline.
    Official code: https://github.com/csdwren/PReNet

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 32.
        num_stages (int): Number of recursions. Default: 6.
        num_resblocks (int): Number of residual blocks. Default: 5.
        recursive_resblock (bool): Whether use recursive reisudal block or not. Default: ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 32,
        num_stages: int = 6,
        num_resblocks: int = 5,
        recursive_resblock: bool = False,
    ) -> None:
        super().__init__()

        self.num_stages = num_stages

        self.feat = nn.Conv2d(in_channels*2, mid_channels, kernel_size=3, stride=1, padding=1)
        self.body = ResBlockBody(mid_channels, num_resblocks, recursive_resblock)
        self.last = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        rainy = x
        outputs = []

        for _ in range(self.num_stages):
            prev_stage_out = outputs[-1] if len(outputs) > 0 else rainy
            x = torch.cat((rainy, prev_stage_out), dim=1)
            x = self.feat(x)
            x = self.body(x)
            x = self.last(x)
            outputs.append(x)

        return outputs


class RecurrentUnit(nn.Module):

    recurrent_units = {
        'GRU': ConvGRU,
        'LSTM': ConvLSTM
    }

    def __init__(self, type: str, planes: int) -> None:
        super().__init__()

        self.model_type = type
        self.model = self.recurrent_units[type](planes, planes, kernel_size=3)

    def forward(
        self,
        x: torch.Tensor,
        hx: Union[torch.Tensor, Tuple[torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:
        if self.model_type == 'LSTM':
            h, c = self.model(x, hx)
            return h, (h, c)
        else:
            h = self.model(x, hx)
            return h, h


@BACKBONES.register_module()
class PReNet(nn.Module):
    """PReNet Network Strcutre

    Paper: Progressive Image Deraining Networks: A Better and Simpler Baseline.
    Official code: https://github.com/csdwren/PReNet

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 32.
        num_stages (int): Number of recursions. Default: 6.
        num_resblocks (int): Number of residual blocks. Default: 5.
        recursive_resblock (bool): Whether use recursive reisudal block or not. Default: ``False``.
    """

    valid_recurrent_units = {'GRU', 'LSTM'}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 32,
        recurrent_unit: str = 'LSTM',
        num_stages: int = 6,
        num_resblocks: int = 5,
        recursive_resblock: bool = False,
    ) -> None:
        super().__init__()

        if recurrent_unit not in self.valid_recurrent_units:
            raise ValueError(f'invalid recurrent unit type {recurrent_unit} for PReNet')

        self.num_stages = num_stages

        self.feat = nn.Conv2d(in_channels*2, mid_channels, kernel_size=3, stride=1, padding=1)
        self.rec_unit = RecurrentUnit(recurrent_unit, mid_channels)
        self.body = ResBlockBody(mid_channels, num_resblocks, recursive_resblock)
        self.last = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        rainy = x
        memory = None
        outputs = []

        for _ in range(self.num_stages):
            prev_stage_out = outputs[-1] if len(outputs) > 0 else rainy
            x = torch.cat((rainy, prev_stage_out), dim=1)
            x = self.feat(x)
            x, memory = self.rec_unit(x, memory)
            x = self.body(x)
            x = self.last(x)
            outputs.append(x)

        return outputs
