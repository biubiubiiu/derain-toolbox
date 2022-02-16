from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mmderain.models.registry import BACKBONES
from mmderain.utils.functools import zip_with_next


class MSFF(nn.Module):
    """Multi-Scale Feature Fusion Block"""

    def __init__(self, in_planes: List[int], planes: int) -> None:
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(x, planes, kernel_size=1, stride=1, padding=0)
            for x in in_planes
        ])
        self.interpolate = partial(F.interpolate, mode='bilinear', align_corners=False)

    def forward(self, x: Sequence[torch.Tensor], output_size: Tuple[int]):
        assert len(self.convs) == len(x)

        outputs = []
        for (conv, feat) in zip(self.convs, x):
            out = conv(feat)  # adjust channels
            out = self.interpolate(input=out, size=output_size)  # adjust size
            outputs.append(out)

        out = torch.stack(outputs, dim=0).sum(dim=0)
        return out


class Rescale(nn.Module):

    def __init__(self, scale_factor: int, upsample: bool) -> None:
        super().__init__()

        self.scale_factor = scale_factor
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return F.max_pool2d(x, kernel_size=2, stride=2)


class EncoderBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int, upsample: bool) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            Rescale(scale_factor=2, upsample=upsample),
            nn.ReLU()
        )

    def forward(
        self,
        x: torch.Tensor,
        msff_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.model(x)
        if msff_feat is not None:
            out = out + msff_feat
        return out


class DecoderBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, upsample: bool) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            Rescale(scale_factor=2, upsample=upsample),
            nn.ReLU()
        )

    def forward(
        self,
        x: torch.Tensor,
        bridge: Optional[torch.Tensor] = None,
        msff_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.model(x)
        if bridge is not None:
            out = out + bridge
        if msff_feat is not None:
            out = out + msff_feat
        return out


@BACKBONES.register_module()
class OUCDNet(nn.Module):
    """OUCDNet Network Structure

    Paper: Exploring Overcomplete Representations for Single Image Deraining Using CNNs
    Official Code: https://github.com/jeya-maria-jose/Derain_OUCD_Net
    """

    def __init__(
        self,
        out_channels: int,
        enc_undercomplete_channels: List[int] = [3, 32, 64, 128, 512, 1024],
        dec_undercomplete_channels: List[int] = [1024, 512, 128, 64, 32, 16],
        enc_overcomplete_channels: List[int] = [3, 32, 64, 128],
        dec_overcomplete_channels: List[int] = [128, 64, 32, 16],
        use_msff: bool = True
    ) -> None:
        super().__init__()

        if enc_undercomplete_channels[0] != enc_overcomplete_channels[0]:
            raise ValueError('Input of two branches should have the same channels')
        if dec_undercomplete_channels[-1] != dec_overcomplete_channels[-1]:
            raise ValueError('Output of two branches should have the same channels')

        self.enc_overcomplete = nn.ModuleList([
            EncoderBlock(in_plane, out_plane, upsample=True)
            for (in_plane, out_plane) in zip_with_next(enc_overcomplete_channels)
        ])
        self.dec_overcomplete = nn.ModuleList([
            DecoderBlock(in_plane, out_plane, upsample=False)
            for(in_plane, out_plane) in zip_with_next(dec_overcomplete_channels)
        ])

        self.enc_undercomplete = nn.ModuleList([
            EncoderBlock(in_plane, out_plane, upsample=False)
            for (in_plane, out_plane) in zip_with_next(enc_undercomplete_channels)
        ])
        self.dec_undercomplete = nn.ModuleList([
            DecoderBlock(in_plane, out_plane, upsample=True)
            for(in_plane, out_plane) in zip_with_next(dec_undercomplete_channels)
        ])

        self.use_msff = use_msff
        if self.use_msff:
            self.msff0 = MSFF(enc_overcomplete_channels[-3:], enc_undercomplete_channels[1])
            self.msff1 = MSFF(dec_overcomplete_channels[-3:], dec_undercomplete_channels[-2])

        self.last = nn.Sequential(
            nn.Conv2d(dec_undercomplete_channels[-1], out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Overcomplete branch
        out = x
        feat_overcomplete = []
        for model in self.enc_overcomplete:
            out = model(out)
            feat_overcomplete.append(out)

        for i, model in enumerate(self.dec_overcomplete):
            bridge = feat_overcomplete[len(self.dec_overcomplete)-i-2] \
                if i < len(self.dec_overcomplete)-1 else None
            out = model(out, bridge)
            feat_overcomplete.append(out)

        # MSFF Fusion
        msff_feat0 = None
        msff_feat1 = None
        if self.use_msff:
            in_size = tuple(x.shape)[2:]
            msff_size = tuple(s // 2 for s in in_size)
            msff_feat0 = self.msff0(feat_overcomplete[:3], msff_size)
            msff_feat1 = self.msff1(feat_overcomplete[3:], msff_size)

        # Undercomplete branch
        out = x
        feat_undercomplete = []
        for i, model in enumerate(self.enc_undercomplete):
            msff = msff_feat0 if i == 0 else None
            out = model(out, msff)
            feat_undercomplete.append(out)

        for i, model in enumerate(self.dec_undercomplete):
            bridge = feat_undercomplete[len(self.dec_undercomplete)-i-2] \
                if i < len(self.dec_undercomplete)-1 else None
            msff = msff_feat1 if i == len(self.dec_undercomplete)-2 else None
            out = model(out, bridge, msff)
            feat_undercomplete.append(out)

        # final fusion
        out = feat_overcomplete[-1] + feat_undercomplete[-1]
        out = self.last(out)
        return out
