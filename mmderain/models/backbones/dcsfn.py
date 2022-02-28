from functools import partial
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from mmderain.models.common import make_layer, sizeof
from mmderain.models.layers import RESCAN_GRU, RESCAN_LSTM, RESCAN_RNN
from mmderain.models.registry import BACKBONES
from mmderain.utils.functools import zip_with_next


def conv3x3(in_chn: int, out_chn: int, bias: bool = True) -> nn.Module:
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv1x1(in_chn: int, out_chn: int, bias: bool = True) -> nn.Module:
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=bias)
    return layer


def upsample(x: Union[torch.Tensor, Iterable[torch.Tensor]], size: List[int]) -> torch.Tensor:
    if all(s == 1 for s in size):
        return x

    if isinstance(x, torch.Tensor):
        x = (x,)

    ret = [F.interpolate(t, size=size, mode='nearest', align_corners=None) for t in x]
    return ret[0] if len(ret) == 1 else ret


def downsample(x: Union[torch.Tensor, Iterable[torch.Tensor]], size: List[int]) -> torch.Tensor:
    if all(s == 1 for s in size):
        return x

    if isinstance(x, torch.Tensor):
        x = (x,)

    ret = [F.interpolate(t, size=size, mode='nearest', align_corners=None) for t in x]
    return ret[0] if len(ret) == 1 else ret


def maxpooling(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return F.max_pool2d(x, scale_factor, scale_factor)


class AlignScale(nn.Module):

    def __init__(self, alignment: str = 'largest') -> None:
        super().__init__()

        valid_alignments = ('largest', 'smallest')
        if alignment not in valid_alignments:
            raise ValueError(f'supported alignments are {valid_alignments},\
                but got alignment={alignment}')

        self.alignment = alignment

    def forward(self, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        sizes = [sizeof(x) for x in tensors]
        if self.alignment == 'largest':
            max_size = max(sizes)
            out = [upsample(x, size=max_size) for x in tensors]
        elif self.alignment == 'smallest':
            min_size = min(sizes)
            out = [downsample(x, size=min_size) for x in tensors]

        out = torch.cat(out, dim=1)
        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int, kernel_size: int = 3) -> None:
        super().__init__()

        if kernel_size == 3:
            self.conv = conv3x3(inplanes, planes, bias=True)
        elif kernel_size == 1:
            self.conv = conv1x1(inplanes, planes, bias=True)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class RecurrentUnit(nn.Module):

    recurrent_units = {
        'GRU': RESCAN_GRU,
        'LSTM': RESCAN_LSTM,
        'RNN': RESCAN_RNN
    }

    def __init__(self, type: str, in_planes: int, planes: int) -> None:
        super().__init__()

        self.model_type = type
        self.model = self.recurrent_units[type](in_planes, planes, kernel_size=3, dilation=1)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, ...]:
        if self.model_type == 'LSTM':
            h, c = self.model(x, hx)
            return h, (h, c)
        else:
            h = self.model(x, hx)
            return h, h


class InnerScaleFusionBlock(nn.Module):

    def __init__(self, planes: int, num_scales: int, num_convs: int) -> None:
        super().__init__()

        self.scales = [2**i for i in range(num_scales)]
        self.scales.reverse()  # descending order

        self.convs = nn.ModuleDict([
            f'scale_{s}',
            make_layer(BasicBlock, num_convs, inplanes=planes, planes=planes, kernel_size=3)
        ] for s in self.scales)

        self.cross_scale_fusions = nn.ModuleDict([
            f'fusion_{sa}_{sb}',
            nn.Sequential(
                AlignScale(alignment='largest'),
                BasicBlock(2*planes, planes, kernel_size=1)
            )
        ] for (sa, sb) in zip_with_next(self.scales))

        self.all_scale_fusion = nn.Sequential(
            AlignScale(alignment='largest'),
            BasicBlock(num_scales*planes, planes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_features = []
        for i, scale in enumerate(self.scales):
            feat = maxpooling(x, scale_factor=scale)

            if i > 0:
                prev_scale = self.scales[i-1]
                fusion = self.cross_scale_fusions[f'fusion_{prev_scale}_{scale}']
                feat = fusion([scaled_features[-1], feat])

            feat = self.convs[f'scale_{scale}'](feat)
            scaled_features.append(feat)

        out = self.all_scale_fusion(scaled_features)
        return out + x


class RecUnitStateFusionBlock(nn.Module):

    def __init__(self, planes: int, n_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            BasicBlock(2*planes, planes, kernel_size=1)
            for _ in range(n_blocks)
        ])

    def forward(
        self,
        hidden_state_a: Union[torch.Tensor, Tuple[torch.Tensor]],
        hidden_state_b: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor]:
        if isinstance(hidden_state_a, torch.Tensor):
            hidden_state_a = (hidden_state_a,)
        if isinstance(hidden_state_b, torch.Tensor):
            hidden_state_b = (hidden_state_b,)

        assert len(hidden_state_a) == len(hidden_state_b)

        out = list()
        for ha, hb, model in zip(hidden_state_a, hidden_state_b, self.blocks):
            out.append(model(torch.cat((ha, hb), dim=1)))

        return out if len(out) > 1 else out[0]


class CrossScaleFusion(nn.Module):

    def __init__(self, planes: int, recurrent_unit: str) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            RecurrentUnit(recurrent_unit, planes, planes)
            for _ in range(6)
        ])

        num_fusion_blocks = 2 if recurrent_unit == 'LSTM' else 1
        self.fusions = nn.ModuleList([
            RecUnitStateFusionBlock(planes, num_fusion_blocks)
            for _ in range(2)
        ])

    def forward(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        """
        NOTE: The direction of the information flow is
        "small scale -> large scale -> small scale",
        which is opposite to what is shown in the paper.

        That's what the authors actually did in their code.
        Never mind, you will get used to it.
        """
        s1_feat, s2_feat, s4_feat = tensors

        # Upward pass
        s4_feat, h_s4 = self.blocks[0](s4_feat)
        s2_feat, h_s2 = self.blocks[1](s2_feat, upsample(h_s4, size=sizeof(s2_feat)))

        fusion = self.fusions[0](
            upsample(h_s4, size=sizeof(s1_feat)),
            upsample(h_s2, size=sizeof(s1_feat)),
        )

        s1_feat, _ = self.blocks[2](s1_feat, fusion)

        # Downward pass
        s1_feat, h_s1 = self.blocks[3](s1_feat)
        s2_feat, h_s2 = self.blocks[4](s2_feat, downsample(h_s1, size=sizeof(s2_feat)))

        fusion = self.fusions[1](
            downsample(h_s1, size=sizeof(s4_feat)),
            downsample(h_s2, size=sizeof(s4_feat)),
        )

        s4_feat, h_s4 = self.blocks[5](s4_feat, fusion)

        return [s1_feat, s2_feat, s4_feat]


class Encoder(nn.Module):

    def __init__(self, planes: int, n_layers: int, n_inner_scales: int, n_inner_convs: int) -> None:
        super().__init__()

        self.models = nn.ModuleList([
            InnerScaleFusionBlock(planes, n_inner_scales, n_inner_convs)
            for _ in range(n_layers)
        ])
        self.fusions = nn.ModuleList([
            BasicBlock((i+2)*planes, planes, kernel_size=1)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = [x]
        out = x

        for model, fusion in zip(self.models, self.fusions):
            feat = model(out)
            features.append(feat)
            out = fusion(torch.cat(features, dim=1))

        return out, features[1:]


class Decoder(nn.Module):

    def __init__(self, planes: int, n_layers: int, n_inner_scales: int, n_inner_convs: int) -> None:
        super().__init__()

        self.models = nn.ModuleList([
            InnerScaleFusionBlock(planes, n_inner_scales, n_inner_convs)
            for _ in range(n_layers)
        ])
        self.fusions = nn.ModuleList([
            BasicBlock((i+2)*planes, planes, kernel_size=1)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, bridges: List[torch.Tensor]) -> torch.Tensor:
        features = [x]

        out = x
        for (model, fusion, bridge) in zip(self.models, self.fusions, bridges):
            feat = model(out + bridge)
            features.append(feat)
            out = fusion(torch.cat(features, dim=1))

        return out


@ BACKBONES.register_module()
class DCSFN(nn.Module):
    """DCSFN Network Structure

    Paper: DCSFN: Deep Cross-scale Fusion Network for Single Image Rain Removal
    Official Code: https://github.com/Ohraincu/DCSFN
    """

    valid_recurrent_units = {'GRU', 'LSTM', 'RNN'}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 20,
        recurrent_unit: str = 'LSTM',
        num_encoder_decoder_layers: int = 16,
        num_inner_scales: int = 4,
        num_inner_convs: int = 4,
    ) -> None:
        super().__init__()

        self.feat = BasicBlock(in_channels, mid_channels, kernel_size=3)

        init_encoder = partial(Encoder, mid_channels, num_encoder_decoder_layers,
                               num_inner_scales, num_inner_convs)
        self.encoder_s1 = init_encoder()
        self.encoder_s2 = init_encoder()
        self.encoder_s4 = init_encoder()

        self.cross_scale_fusion = CrossScaleFusion(mid_channels, recurrent_unit)

        init_decoder = partial(Decoder, mid_channels, num_encoder_decoder_layers,
                               num_inner_scales, num_inner_convs)
        self.decoder_s1 = init_decoder()
        self.decoder_s2 = init_decoder()
        self.decoder_s4 = init_decoder()

        self.align_scale = AlignScale(alignment='largest')

        self.last = nn.Sequential(
            BasicBlock(3*mid_channels, out_channels, kernel_size=3),
            conv3x3(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feat(x)

        out_s1 = out
        out_s2 = maxpooling(out, scale_factor=2)
        out_s4 = maxpooling(out, scale_factor=4)

        out_s1, s1_feats = self.encoder_s1(out_s1)
        out_s2, s2_feats = self.encoder_s2(out_s2)
        out_s4, s4_feats = self.encoder_s4(out_s4)

        out_s1, out_s2, out_s4 = self.cross_scale_fusion(out_s1, out_s2, out_s4)

        out_s1 = self.decoder_s1(out_s1, s1_feats)
        out_s2 = self.decoder_s2(out_s2, s2_feats)
        out_s4 = self.decoder_s4(out_s4, s4_feats)

        out = self.align_scale([out_s1, out_s2, out_s4])
        out = self.last(out)

        return x - out
