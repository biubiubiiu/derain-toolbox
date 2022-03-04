import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from mmderain.models.registry import BACKBONES


class DCM(nn.Module):
    """Dilated Convolutional Module"""

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.fusion = nn.Conv2d(5 * planes, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]

        out = F.relu(self.conv1(x))
        features.append(out)
        out = F.relu(self.conv2(out))
        features.append(out)

        out = F.relu(self.conv3(x))
        features.append(out)
        out = F.relu(self.conv4(out))
        features.append(out)

        out = torch.cat(features, dim=1)
        out = F.relu(self.fusion(out))
        return out


class sGCN(nn.Module):
    """Spatial GCN Module"""

    def __init__(self, in_planes: int, planes: int) -> None:
        super().__init__()

        self.theta = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)
        self.nu = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)
        self.ksi = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(planes, in_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        theta = self.theta(x)
        theta = rearrange(theta, 'b c h w -> b c (h w)')

        nu = self.nu(x)
        nu = rearrange(nu, 'b c h w -> b (c h w)')
        nu = F.softmax(nu, dim=-1)
        nu = rearrange(nu, 'b (c h w) -> b c (h w)', h=h, w=w)

        ksi = self.ksi(x)
        ksi = rearrange(ksi, 'b c h w -> b (c h w)')
        ksi = F.softmax(ksi, dim=-1)
        ksi = rearrange(ksi, 'b (c h w) -> b c (h w)', h=h, w=w)

        ksi_T = rearrange(ksi, 'b c hw -> b hw c')

        F_s = torch.matmul(nu, ksi_T)  # b*c*c
        AF_s = torch.matmul(F_s, theta)
        AF_s = rearrange(AF_s, 'b c (h w) -> b c h w', h=h, w=w)

        F_sGCN = self.conv(AF_s)
        return x + F_sGCN


class cGCN(nn.Module):
    """Channel GCN Module"""

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.zeta = nn.Conv2d(planes, planes // 2, kernel_size=1, stride=1, padding=0)
        self.kappa = nn.Conv2d(planes, planes // 4, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv1d(planes // 2, planes // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(planes // 4, planes // 4, kernel_size=1, stride=1, padding=0)

        self.conv_expand = nn.Conv2d(planes // 4, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape

        c2 = c // 2
        c4 = c // 4

        zeta = self.zeta(x)
        zeta = rearrange(zeta, 'b c2 h w -> b c2 (h w)')
        zeta_T = rearrange(zeta, 'b c2 hw -> b hw c2')

        kappa = self.kappa(x)
        kappa = rearrange(kappa, 'b c4 h w -> b c4 (h w)')

        F_c = torch.matmul(kappa, zeta_T)
        F_c = rearrange(F_c, 'b c4 c2 -> b (c4 c2)')
        F_c = F.softmax(F_c, dim=-1)
        F_c = rearrange(F_c, 'b (c4 c2) -> b c4 c2', c4=c4, c2=c2)

        F_c = rearrange(F_c, 'b c4 c2 -> b c2 c4')
        F_c = F_c + self.conv1(F_c)
        F_c = F.relu(F_c)
        F_c = rearrange(F_c, 'b c2 c4 -> b c4 c2')
        F_c = self.conv2(F_c)
        F_c = rearrange(F_c, 'b c4 c2 -> b c2 c4')

        F_c = torch.matmul(zeta_T, F_c)
        F_c = rearrange(F_c, 'b (h w) c4 -> b c4 h w', h=h, w=w)

        F_cGCN = self.conv_expand(F_c)
        return x + F_cGCN


class BasicBlock(nn.Module):

    def __init__(self, planes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            sGCN(planes, planes // 2),
            DCM(planes),
            cGCN(planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


@BACKBONES.register_module()
class DualGCN(nn.Module):
    """DualGCN Network Structure

    Paper: https://ojs.aaai.org/index.php/AAAI/article/view/16224
    Official Code: https://xueyangfu.github.io/paper/2021/AAAI/code.zip

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features. Default: 72.
        num_blocks (int): Depth of network. Default 11.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 72,
        num_blocks: int = 11
    ) -> None:

        super().__init__()

        self.feat0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.feat1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        encoder = [BasicBlock(mid_channels)] * (num_blocks // 2)
        decoder = [
            nn.Sequential(
                nn.Conv2d(mid_channels*2, mid_channels, kernel_size=1, stride=1, padding=0),
                BasicBlock(mid_channels)
            )
        ] * (num_blocks // 2)

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        self.bottleneck = BasicBlock(mid_channels) if num_blocks % 2 > 0 else None

        self.recons0 = nn.Conv2d(mid_channels*2, mid_channels, kernel_size=3, stride=1, padding=1)
        self.recons1 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat0 = self.feat0(x)
        feat1 = self.feat1(feat0)

        feat = feat1
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)

        if self.bottleneck is not None:
            feat = self.bottleneck(feat)

        for i, block in enumerate(self.decoder):
            feat = torch.cat([feat, encoder_features[-i-1]], dim=1)
            feat = block(feat)

        out = F.relu(self.recons0(torch.cat([feat, feat1], dim=1)))
        out = self.recons1(out + feat0)

        return x + out
