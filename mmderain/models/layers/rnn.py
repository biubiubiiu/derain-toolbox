from typing import Optional, Tuple

import torch
from torch import nn


class ConvRNN(nn.Module):
    r"""A Elman RNN cell.

    .. math::

        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

    Args:
        input_channel (int): Channel number of inputs.
        output_channel (int): Channle number of outputs.
        kernel_size (int): Size of the convolving kernel.
        padding (int): Padding added to all four sides of the input. Default: 0
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel_size: int,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.output_channel = output_channel

        self.conv_x = nn.Conv2d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.conv_h = nn.Conv2d(
            in_channels=output_channel,
            out_channels=output_channel,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hx is None:
            b, _, h, w = x.shape
            hx = torch.zeros((b, self.output_channel, h, w)).to(x)

        h_next = self.conv_x(x) + self.conv_h(hx)
        return h_next


class ConvGRU(nn.Module):
    r"""A gated recurrent unit GRU cell.

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        input_channel (int): Channel number of inputs.
        output_channel (int): Channle number of outputs.
        kernel_size (int): Size of the convolving kernel.
        padding (int): Padding added to all four sides of the input. Default: 0
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel_size: int,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.output_channel = output_channel

        self.conv0 = nn.Conv2d(
            in_channels=input_channel + output_channel,
            out_channels=2*output_channel,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.conv1 = nn.Conv2d(
            in_channels=input_channel + output_channel,
            out_channels=output_channel,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if hx is None:
            b, _, h, w = x.shape
            hx = torch.zeros((b, self.output_channel, h, w)).to(x)

        combined = torch.cat([x, hx], dim=1)

        combined_conv = self.conv0(combined)
        cc_r, cc_z = torch.split(combined_conv, self.output_channel, dim=1)
        r = torch.sigmoid(cc_r)
        z = torch.sigmoid(cc_z)

        input_reset = torch.cat([x, r*hx], dim=1)
        n = torch.tanh(self.conv1(input_reset))

        h_next = (1 - z) * n + z * hx
        return h_next


class ConvLSTM(nn.Module):
    r"""A long short-term memory LSTM cell.

    .. math::
        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}
    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        input_channel (int): Channel number of inputs.
        output_channel (int): Channle number of outputs.
        kernel_size (int): Size of the convolving kernel.
        padding (int): Padding added to all four sides of the input. Default: 0
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel_size: int,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.output_channel = output_channel

        self.conv = nn.Conv2d(
            in_channels=input_channel + output_channel,
            out_channels=4*output_channel,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            b, _, h, w = x.shape
            zeros = torch.zeros((b, self.output_channel, h, w)).to(x)
            hx = (zeros, zeros)

        h_cur, c_cur = hx
        combined = torch.cat([x, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.output_channel, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
