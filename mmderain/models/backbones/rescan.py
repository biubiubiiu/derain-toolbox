from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from mmcv.runner import load_checkpoint
from torch import nn

from mmderain.models.layers import RESCAN_GRU, RESCAN_LSTM, RESCAN_RNN, SELayer
from mmderain.models.registry import BACKBONES
from mmderain.utils import get_root_logger


class RecurrentUnit(nn.Module):

    recurrent_units = {
        "GRU": RESCAN_GRU,
        "LSTM": RESCAN_LSTM,
        "RNN": RESCAN_RNN
    }

    def __init__(
        self,
        type: str,
        in_planes: int,
        planes: int,
        kernel_size: int,
        dilation: int
    ) -> None:
        super().__init__()

        self.model_type = type
        self.model = self.recurrent_units[type](in_planes, planes, kernel_size, dilation)
        self.se = SELayer(planes, reduction=4)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]
    ) -> Tuple[torch.Tensor, ...]:
        if self.model_type == "LSTM":
            h, c = self.model(x, hx)
            h = self.act(self.se(h))
            return h, (h, c)
        else:
            h = self.model(x, hx)
            h = self.act(self.se(h))
            return h, h


@ BACKBONES.register_module()
class RESCAN(nn.Module):
    """RESCAN Network Structure

    Paper: Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining.
    Official Code: https://github.com/XiaLiPKU/RESCAN
    """

    valid_recurrent_units = {"GRU", "LSTM", "RNN"}
    valid_prediction_types = {"Additive", "Full"}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 24,
        num_stages: int = 4,
        depth: int = 5,
        recurrent_unit: str = "GRU",
        prediction_type="Full",
    ) -> None:

        super().__init__()

        if recurrent_unit not in self.valid_recurrent_units:
            raise KeyError(f"invalid recurrent unit type {recurrent_unit} for RESCAN")

        if prediction_type not in self.valid_prediction_types:
            raise ValueError(
                f"prediction_type must be one of {self.valid_prediction_types},\
                but got prediction_type={prediction_type}"
            )

        self.num_stages = num_stages
        self.prediction_type = prediction_type

        self.block = partial(RecurrentUnit, recurrent_unit)

        self.recurrent_models = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                block = self.block(in_channels, mid_channels, kernel_size=3, dilation=1)
            else:
                block = self.block(mid_channels, mid_channels, kernel_size=3, dilation=2 ** (d - 1))
            self.recurrent_models.append(block)

        self.output_layer = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            SELayer(mid_channels, reduction=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ori = x

        bg_estimates = []
        memories = [None] * len(self.recurrent_models)
        prev_bg_estimate = x
        prev_rain_estimate = torch.zeros_like(x)

        for s in range(self.num_stages):
            x = prev_bg_estimate

            states = []
            for model, state in zip(self.recurrent_models, memories):
                x, st = model(x, state)
                states.append(st)

            memories = states.copy()

            x = self.output_layer(x)  # estimation of rain layer
            if self.prediction_type == "Additive" and s > 0:
                # accumulate previous rain estimates
                x = x + torch.clone(prev_rain_estimate)

            prev_rain_estimate = x
            prev_bg_estimate = ori - prev_rain_estimate
            bg_estimates.append(prev_bg_estimate)

        # The paper use MSE loss between rain layer and rain estimation,
        # which holds that \sum (R-R_i)^2 = \sum(O-B-(O-B_i))^2 = \sum (B-B_i)^2
        # For simplicity, returns estimates of backgrounds instead of rains here
        return bg_estimates

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
