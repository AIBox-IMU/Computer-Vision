from typing import Optional
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))

        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels ,kernel_size=7,reduction=4):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super(FusionConv, self).__init__()
        dim = int(out_channels )
        channels = in_channels
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)  # 卷积模块1
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

        self.conv_7x71 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.conv_9x91 = nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4)
        self.conv_11x111 = nn.Conv2d(dim, dim, kernel_size=11, stride=1, padding=5)
        h_kernel_size: int = 7 # 水平卷积核大小，默认为11
        v_kernel_size: int = 7
        self.h_conv7 = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)  # 水平卷积模块

        self.v_conv7 = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)  # 垂直卷积模块
        h_kernel_size: int = 9  # 水平卷积核大小，默认为11
        v_kernel_size: int = 9
        self.h_conv9 = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                (0, h_kernel_size // 2), groups=channels,
                                norm_cfg=None, act_cfg=None)  # 水平卷积模块

        self.v_conv9 = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)  # 垂直卷积模块
        h_kernel_size: int = 11  # 水平卷积核大小，默认为11
        v_kernel_size: int = 11
        self.h_conv11 = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                (0, h_kernel_size // 2), groups=channels,
                                norm_cfg=None, act_cfg=None)  # 水平卷积模块

        self.v_conv11 = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)  # 垂直卷积模块
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)  # 卷积模块2
        self.spatial_attention = SpatialAttentionModule(dim)
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x_fused):
        x_fused_c = x_fused * self.channel_attention(x_fused)
        avg = self.conv1(self.avg_pool(x_fused))
        x_7x7 = self.h_conv7(avg)
        x_7x7 = self.v_conv7(x_7x7)
        x_7x7 = self.conv2(x_7x7)
        x_9x9 = self.h_conv9(avg)
        x_9x9 = self.v_conv9(x_9x9)
        x_9x9 = self.conv2(x_9x9)
        x_11x11 = self.h_conv11(avg)
        x_11x11 = self.v_conv11(x_11x11)
        x_11x11 = self.conv2(x_11x11)
        x_fused_s = x_7x7 + x_9x9 + x_11x11
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        x_out = self.up(x_fused_s + x_fused_c)
        return x_out


class LSAMA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSAMA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, last=False):
        x_fused = self.fusion_conv(x1)
        return x_fused
