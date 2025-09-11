import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init


class GatedFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(GatedFeatureFusion, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=0)
    def split_tensor(self, input_tensor):
        batch_size = input_tensor.size(0)
        split_tensors = []
        for i in range(batch_size):
            split_tensor = input_tensor[i]
            split_tensors.append(split_tensor)
        return split_tensors
    def forward(self, d1, d2):

        assert d1.shape == d2.shape, "d1 和 d2 的形状必须相同"
        gap_tensor1 = self.gap(d1)
        gap_tensor2 = self.gap(d2)
        gap_weight1 = self.sigmoid(gap_tensor1)
        gap_weight2 = self.sigmoid(gap_tensor2)
        gap_weight = torch.cat([gap_weight1, gap_weight2], dim=1)
        d = torch.cat([d1, d2], dim=1)
        gap_weight = self.softmax(gap_weight)
        result = d * gap_weight
        d1, d2 = torch.chunk(result, 2, dim=1)
        return d1, d2

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ResidualBlock_noBN(nn.Module):
     def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

     def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class MDFF(nn.Module):
    def __init__(self, nf=64):
        super(MDFF, self).__init__()
        dim = nf
        self.conv1 = ResidualBlock_noBN(nf)
        self.conv2 = self.conv1
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)
        self.convf2 = self.convf1
        self.scale = nf ** -0.5
        self.norm_s = LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)
        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.gated1 = nn.Sequential(
            nn.Conv2d(nf, nf, 1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.gated2 = nn.Sequential(
            nn.Conv2d(nf, nf, 1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.mix = GatedFeatureFusion(dim) # Module
        # initialization
        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_p, x_n):
        b, c, h, w = x_p.shape

        x_p_ = self.conv1(x_p)
        x_n_ = self.conv2(x_n)
        x_int = torch.cat((x_p_, x_n_), dim=1)
        x_xp,x_xn  = self.mix(x_p, x_n)
        x_int =  x_xp +  x_xn
        x_xp = x_xp.view(b, c, -1)
        x_xn = x_xn.view(b, c, -1)
        k_p = self.v1(x_p).view(b,c,-1).permute(0, 2, 1)
        k_n = self.v2(x_n).view(b,c,-1).permute(0, 2, 1)
        att1 = torch.bmm(x_xp, k_n) * self.scale
        att2 = torch.bmm(x_xn, k_p) * self.scale
        z_pn = torch.bmm(torch.softmax(att1, dim=-1), k_n.permute(0, 2, 1)).view(b, c, h, w)
        z_np = torch.bmm(torch.softmax(att2, dim=-1), k_p.permute(0, 2, 1)).view(b, c, h, w)
        W_p = self.gated1(z_pn + x_p_)
        Y_p = W_p * z_pn + (1-W_p) * x_p_
        W_n = self.gated1(z_np + x_n_)
        Y_n = W_n * z_np + (1 - W_n) * x_n_
        Y = self.unclustering(torch.cat([x_xp.view(b, c, h, w), x_xn.view(b, c, h, w)], dim=1))
        fused_feature = (Y + Y_p + Y_n) / 3.0 + x_int
        return fused_feature

