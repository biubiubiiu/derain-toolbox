from typing import List, Optional, Tuple, Union

import torch
from mmcv.runner import load_checkpoint
from torch import nn

from mmderain.models.common import make_layer
from mmderain.models.layers import ConvGRU, ConvLSTM
from mmderain.models.registry import BACKBONES
from mmderain.utils import get_root_logger


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
    """PRN Network Structure

    This is the simplified version of PReNet, where recurrent unit is removed
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
        elif pretrained is not None:
            raise TypeError(
                f'"pretrained" must be a str or None. ' f"But received {type(pretrained)}."
            )


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

    Paper: Progressive Image Deraining Networks: A Better and Simpler Baseline
    Official code: https://github.com/csdwren/PReNet
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
            raise KeyError(f'invalid recurrent unit type {recurrent_unit} for PReNet')

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
