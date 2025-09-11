# 特征交互
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class MixedDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixedDilatedConv, self).__init__()
        self.dilatedconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.dilatedconv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=3, padding=3)
        self.dilatedconv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=5, padding=5)
        self.dilatedconv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=7, padding=7)

    def forward(self, x):
        out1 = self.dilatedconv1(x)
        out2 = self.dilatedconv2(x)
        out3 = self.dilatedconv3(x)
        out4 = self.dilatedconv4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)

class FeatureChange(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prob = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.b = 0.5
    def forward(self, x1t, x2t):
        P1, P2 = self.sigmoid(self.prob(x1t)), self.sigmoid(self.prob(x2t))
        P1 = P1 * (1 - self.b) + self.b
        x = P1 * x1t + (1 - P1) * P2 * x2t
        return x

class DSFFT(nn.Module):
    def __init__(self, channels):
        super(DSFFT, self).__init__()
        self.mixdilatedconv = MixedDilatedConv(channels, channels)
        self.DSFFT = FeatureChange(channels)
        self.pimMixedconv = FeatureChange(channels*4)
        self.depthwise_conv = nn.Conv2d(in_channels=channels*4, out_channels=channels, kernel_size=3,
            stride=1,
            padding=1,
            groups=4
        )
        drop_rate = 0.2
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(self, x1, x2):
        shortcut1 = x1
        shortcut2 = x2
        x1 = self.mixdilatedconv(x1)
        x2 = self.mixdilatedconv(x2)
        x1_C = self.DSFFT(shortcut1, shortcut2)
        x1_d = self.pimMixedconv(x1, x2)
        x1 = self.depthwise_conv(x1_d)
        x1 = x1/ 4 + x1_C
        return x1
