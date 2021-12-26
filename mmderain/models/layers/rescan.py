# Modified from RESCAN official code: https://github.com/XiaLiPKU/RESCAN
# ----------------------------------------------------------------------
# There're subtle differences between standard recurrent units and
# the ones used by the authors, as they use convolutions with different
# dilation rates to process input and hidden states

import torch
from torch import nn


class RESCAN_RNN(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel_size, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel_size - 1) / 2)
        self.conv_x = nn.Conv2d(inp_dim, oup_dim, kernel_size, padding=pad_x, dilation=dilation)

        pad_h = int((kernel_size - 1) / 2)
        self.conv_h = nn.Conv2d(oup_dim, oup_dim, kernel_size, padding=pad_h)

    def forward(self, x, h=None):
        if h is None:
            h = torch.tanh(self.conv_x(x))
        else:
            h = torch.tanh(self.conv_x(x) + self.conv_h(h))

        return h


class RESCAN_GRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel_size, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel_size - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel_size, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel_size, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel_size, padding=pad_x, dilation=dilation)

        pad_h = int((kernel_size - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel_size, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel_size, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel_size, padding=pad_h)

    def forward(self, x, h=None):
        if h is None:
            z = torch.sigmoid(self.conv_xz(x))
            f = torch.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = torch.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n

        return h


class RESCAN_LSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel_size, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel_size - 1) / 2)
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel_size, padding=pad_x, dilation=dilation)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel_size, padding=pad_x, dilation=dilation)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel_size, padding=pad_x, dilation=dilation)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel_size, padding=pad_x, dilation=dilation)

        pad_h = int((kernel_size - 1) / 2)
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel_size, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel_size, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel_size, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel_size, padding=pad_h)

    def forward(self, x, pair=None):
        if pair is None:
            i = torch.sigmoid(self.conv_xi(x))
            o = torch.sigmoid(self.conv_xo(x))
            j = torch.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            h, c = pair
            f = torch.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = torch.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = torch.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = torch.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * torch.tanh(c)

        return (h, c)
