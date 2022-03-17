from torch import nn


class SELayer(nn.Module):
    """Sequeeze-and-Excitation Layer

    Used in SENet.

    The implementation is taken from https://github.com/moskomule/senet.pytorch

    Args:
        channel (int): Channel number of inputs and outputs.
        reduction (int): Reduction ratio. Default: 16.
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayer_Modified(nn.Module):
    """Customized Sequeeze-and-Excitation Layer

    Args:
        channel (int): Channel number of inputs and outputs.
        reduction (int): Reduction ratio. Default: 16.
        bias (bool): Use bias in linear layers or not. Default: False.
        act (nn.Module): Activation function after the first linear layer. Default: nn.ReLU().
    """

    def __init__(self, channel, reduction=16, bias=False, act=nn.ReLU(inplace=True)):
        super(SELayer_Modified, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=bias),
            act,
            nn.Linear(channel // reduction, channel, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
