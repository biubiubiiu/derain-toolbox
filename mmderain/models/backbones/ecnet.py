from typing import List, Optional, Tuple

import torch
from torch import nn

from mmcv.runner import BaseModule
from mmderain.models.layers import RLCN, ConvLSTM
from mmderain.models.registry import BACKBONES


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, in_channels: int, out_channels: int, use_shortcut: bool) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act_last = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        out = out + self.shortcut(x)
        out = self.act_last(out)
        return out


class ResidualRecurrentBlock(nn.Module):
    """Residual Recurrent Block"""

    def __init__(self, in_channels: int, out_channels: int, use_shortcut: bool) -> None:
        super().__init__()

        self.model0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.lstm = ConvLSTM(out_channels, out_channels, kernel_size=3)
        self.model1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act_last = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor,
                memory: Optional[Tuple[torch.Tensor]] = None) -> Tuple[List[torch.Tensor]]:
        out = self.model0(x)
        h, c = self.lstm(out, memory)
        out = h
        out = self.model1(out)
        out = out + self.shortcut(x)
        out = self.act_last(out)
        return out, (h, c)


class MaskGAM(nn.Module):

    def __init__(self, in_planes: int, mid_planes: int, out_planes: int) -> None:
        super().__init__()

        self.wx = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_planes)
        )
        self.wy = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_planes)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.Sigmoid()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor]:
        out1 = self.wx(x)
        out2 = self.wy(y)
        out = self.act(out1+out2)
        att = self.psi(out)
        out = x*att
        return out, att


class UNetDownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 downsample: bool, use_shortcut: bool) -> None:
        super().__init__()

        models = [ResidualBlock(in_channels, out_channels, use_shortcut)]
        if downsample:
            models.insert(0, nn.MaxPool2d(kernel_size=2, stride=2))

        self.model = nn.Sequential(*models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetRecurrentDownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 downsample: bool, use_shortcut: bool) -> None:
        super().__init__()

        self.downsample = downsample
        self.model = ResidualRecurrentBlock(in_channels, out_channels, use_shortcut)

        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor,
                memory: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.Tensor]:
        out = self.pool(x) if self.downsample else x
        out, mem = self.model(out, memory)
        return out, mem


class UNetUpBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, use_shortcut: bool = True) -> None:
        super().__init__()

        self.model0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.model1 = ResidualBlock(out_channels, out_channels, use_shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model1(self.model0(x))
        return out


class UNetGAMUpBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int,
                 att_channels: int, use_shortcut: bool = True) -> None:
        super().__init__()

        self.model0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.gam = MaskGAM(out_channels, mid_channels, att_channels)
        self.model1 = ResidualBlock(out_channels, out_channels, use_shortcut)
        self.model2 = ResidualBlock(out_channels, out_channels, use_shortcut)

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> Tuple[torch.Tensor]:
        feat = self.model0(x)
        out, att = self.gam(feat, bridge)
        out = self.model1(feat) + self.model2(out)
        return out, att


@BACKBONES.register_module()
class RainEncoder(BaseModule):
    """Rain-to-Rain Network Structure

    Paper: Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM
    Official Code: https://github.com/Yizhou-Li-CV/ECNet

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 32.
        depth (int): Depth of UNet. Default: 4.
        init_cfg: (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 32,
        depth: int = 4,
        init_cfg: Optional[dict] = None
    ) -> None:
        super().__init__(init_cfg)

        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        down_path = []
        up_path = []

        prev_channels = mid_channels
        for i in range(depth):
            use_shortcut = i > 0
            downsample = i > 0
            down_path.append(UNetDownBlock(prev_channels, (2**i)*mid_channels,
                                           downsample=downsample, use_shortcut=use_shortcut))
            prev_channels = (2**i)*mid_channels

        for i in reversed(range(depth-1)):
            up_path.append(UNetUpBlock(prev_channels, (2**i)*mid_channels))
            prev_channels = (2**i)*mid_channels

        self.down_path = nn.Sequential(*down_path)
        self.up_path = nn.Sequential(*up_path)

        self.conv_last = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        out = self.conv_input(x)
        embedding = self.down_path(out)
        out = embedding
        out = self.up_path(out)
        out = self.conv_last(out)
        return out, embedding


@BACKBONES.register_module()
class ECNet(BaseModule):
    """ECNet Network Structure

    Paper: Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM
    Official Code: https://github.com/Yizhou-Li-CV/ECNet

    Args:
        in_out_channels (int): Channel number of inputs and outputs.
        mid_channels (int): Channel number of intermediate features. Default: 32
        depth (int): Depth of UNet. Default: 4
        n_iters (int): Number of recustions. This property would be ignore if `use_rnn`
            is set to ``False``. Default: 6
        lcn_window_size (int): Window size in lcn calculation. Default: 9
        use_rnn (int): Whether use recurrent unit or not. If ``True``, LayerLSTM would
            be integrated. Default: True
        init_cfg: (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        in_out_channels: int,
        mid_channels: int = 32,
        depth: int = 4,
        n_iters: int = 6,
        lcn_window_size: int = 9,
        use_rnn: bool = True,
        init_cfg: Optional[dict] = None
    ) -> None:
        super().__init__(init_cfg)

        self.n_iters = n_iters
        self.use_rnn = use_rnn
        self.lcn_window_size = lcn_window_size

        in_channels = in_out_channels*3 if use_rnn else in_out_channels*2
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        down_block = UNetRecurrentDownBlock if use_rnn else UNetDownBlock
        prev_channels = mid_channels
        for i in range(depth):
            use_shortcut = i > 0
            downsample = i > 0
            self.down_path.append(down_block(prev_channels, (2**i)*mid_channels,
                                             downsample=downsample, use_shortcut=use_shortcut))
            prev_channels = (2**i)*mid_channels

        for i in reversed(range(depth-1)):
            self.up_path.append(UNetGAMUpBlock(prev_channels, (2**i)*mid_channels,
                                               mid_channels=max((2**(i-1)), 1)*mid_channels,
                                               att_channels=1))
            prev_channels = (2**i)*mid_channels

        self.conv_last = nn.Conv2d(mid_channels, in_out_channels,
                                   kernel_size=1, stride=1, padding=0)


    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        rain_residuals = []
        embeddings = []
        attention_maps = []

        lcn, _, _ = RLCN((x+1)/2, self.lcn_window_size)
        lcn = torch.clamp(lcn, min=0.0, max=1.0)

        rain_last_stage = None
        memories = [None for _ in range(4)]

        n_recursions = 1 if not self.use_rnn else self.n_iters
        for _ in range(n_recursions):
            if self.use_rnn:
                prev_out = x if rain_last_stage is None else rain_last_stage
                inp = torch.cat([x, prev_out, lcn], dim=1)
            else:
                inp = torch.cat([x, lcn], dim=1)

            out = self.conv_input(inp)
            encoder_features = []
            for model in self.down_path:
                if self.use_rnn:
                    mem_prev = memories.pop(0)
                    out, mem = model(out, mem_prev)
                    memories.append(mem)
                else:
                    out = model(out)

                encoder_features.append(out)

            embedding = encoder_features.pop()  # remove feature from bottleneck
            out = embedding

            for model in self.up_path:
                out, psi = model(out, encoder_features.pop())

            out = self.conv_last(out)

            rain_residuals.append(out)
            embeddings.append(embedding)
            attention_maps.append(psi)
            rain_last_stage = out

        derains = [x - res for res in rain_residuals]  # apply residual
        return derains, embeddings, attention_maps
