import torch
import torch.nn as nn
import torch.nn.functional as F
from LSAMA import LSAMA
from DSFFT import DSFFT
from MDFF import MDFF

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class IFEBlock(nn.Module):
    def __init__(self, dim, drop_rate=0.2, mode="LN"):
        super().__init__()
        self.depthconv1 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.depthconv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.depthconv3 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        if mode == "BN":
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.norm = LayerNorm(dim, data_format="channels_first")
        self.pointconv1 = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pointconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()
    def forward(self, x):
        shortcut = x
        x = (self.depthconv1(x) + self.depthconv2(x) + self.depthconv3(x)) / 3
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.pointconv1(x)
        x = self.gelu(x)
        x = self.grn(x)
        x = self.pointconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.drop_path(x)
        return x

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)  # 定义平均池化层

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return  edge

class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()

        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(  # 定义输出卷积层
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)
        out = mid
        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)
            edge = self.edge_enhance[i](mid)
            out = torch.cat([out,edge], dim=1)
        out = self.out_conv(out)
        return out

class PatchEmb(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        self.GenPatch = nn.Conv2d(3, dim // 4, kernel_size=4, stride=4)
        self.ln = nn.BatchNorm2d(dim // 4)
        self.GenPatch2 = nn.Conv2d(
            dim // 4, dim, kernel_size=patch_size // 4, stride=patch_size // 4
        )
        self.conv_1x1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=1)
    def forward(self, x):
        x = self.GenPatch(x)
        x = self.ln(x)
        x = self.GenPatch2(x)
        return x

class FEStage(nn.Module):
    def __init__(self, dim, n1=4):
        super().__init__()
        self.n1 = n1
        self.blocks = nn.ModuleList([IFEBlock(dim) for _ in range(n1)])
        self.checks = nn.ModuleList([DSFFT(dim) for _ in range(n1)])
        self.attentions = LSAMA(in_channels=dim, out_channels=dim).cuda()
        self.MEEM = MEEM(in_dim=dim, hidden_dim=32)
    def forward(self, x1, x2):
        diffList = torch.tensor(0.0, dtype=x1.dtype, device=x1.device)
        shortcutx1 = x1
        shortcutx2 = x2
        for num in range(0, self.n1):
            chk = self.checks[num]
            blk = self.blocks[num]
            x1, x2 = blk(x1), blk(x2)
            x1w = self.WindowMaskSimple(x1)
            diff = chk(x1, x1w) - x1
            diffList += torch.mean(diff * diff)
            x2w = self.NoiseSimple(x2)
            diff = chk(x2, x2w) - x2
            diffList += torch.mean(diff * diff)
            x1, x2 = chk(x1, x2), chk(x2, x1)
            if num == 0:
                t1 = x1
                t2 = x2
            elif num == (self.n1-1):
                x1 = (x1 + t1)/2.0
                x2 = (x2 + t2)/2.0


        out1, out2 = self.attentions(shortcutx1),self.attentions(shortcutx2)
        edge1, edge2 = self.MEEM(shortcutx1), self.MEEM(shortcutx2)
        x1 = x1 + out1 * 0.5 + edge1 + shortcutx1
        x2 = x2 + out2 * 0.5 + edge2 + shortcutx2
        return x1, x2, diffList / self.n1 / 2

    def WindowMaskSimple(self, x, drop_prob=0.5):
        keep_prob = 1 - drop_prob
        random_tensor = torch.rand(x.shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor + keep_prob
        random_tensor = random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor

        shape = (x.shape[0], 1, 1, x.shape[3])
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor + keep_prob
        random_tensor = random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
    def NoiseSimple(self, x, drop_prob=0.5):
        random_tensor = torch.rand(x.shape, dtype=x.dtype, device=x.device)
        random_tensor = (random_tensor * 2 - 1) * drop_prob + 1
        return x * random_tensor

class Repatch(nn.Module):
    def __init__(self, in_ch, patch_size):
        super().__init__()
        self.patchup = nn.ConvTranspose2d(
            in_ch, 2, kernel_size=patch_size, stride=patch_size
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, a, b):
        x1, x2 = self.patchup(a), self.patchup(b)
        x1P, x2P = self.softmax(x1), self.softmax(x2)
        spA = torch.zeros_like(x1P)
        spA[:, 1, :, :] = (
            x1P[:, 1, :, :] * x2P[:, 0, :, :] + x1P[:, 0, :, :] * x2P[:, 1, :, :]
        )
        spA[:, 0, :, :] = (
            x1P[:, 0, :, :] * x2P[:, 0, :, :] + x1P[:, 1, :, :] * x2P[:, 1, :, :]
        )
        return spA


class PMFFM(nn.Module):
    def __init__(self, dim, k, m):
        super().__init__()
        self.k = k
        self.m = m
        self.prob = nn.Linear(dim // k, m)
        self.softmax = nn.Softmax(dim=4)
        self.result = nn.Conv2d(dim // k * m, dim // k * m, kernel_size=1, groups=m)
        self.d1line = torch.tensor(
            [1] * (self.m // 2) + [1] * (self.m // 4) + [-1] * (self.m // 4),
            device=torch.device("cuda"),
        ).reshape([1, 1, self.m, 1, 1, 1])
        self.d2line = torch.tensor(
            [1] * (self.m // 2) + [-1] * (self.m // 4) + [1] * (self.m // 4),
            device=torch.device("cuda"),
        ).reshape([1, 1, self.m, 1, 1, 1])

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        m = x1 + x2
        m = m.view([B, self.k, C // self.k, H, W])
        m = m.permute(0, 1, 3, 4, 2).contiguous()
        prob = self.prob(m)
        prob = self.softmax(prob)

        d1, d2 = x1, x2
        d1 = d1.reshape([B, self.k, 1, C // self.k, H, W]).repeat(1, 1, self.m, 1, 1, 1)
        d2 = d2.reshape([B, self.k, 1, C // self.k, H, W]).repeat(1, 1, self.m, 1, 1, 1)
        d = d1 * self.d1line + d2 * self.d2line

        d = d.reshape(B * self.k, self.m * C // self.k, H, W)
        result = self.result(d)
        result = result.reshape(B, self.k, self.m, C // self.k, H, W)

        prob = prob.unsqueeze(4)
        result = result.permute(0, 1, 4, 5, 3, 2).contiguous()
        out = result * prob
        out = torch.sum(out, dim=5)

        out = out.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        return out
class DCIBCD(nn.Module):
    def __init__(self,drop_rate=0.2):
        super().__init__()
        patch_size = 8
        dim = 256
        n1 = n2 = 1
        self.PE = PatchEmb(dim=dim, patch_size=patch_size)
        self.FES = FEStage(dim, n1)
        self.repatch = Repatch(in_ch=dim, patch_size=patch_size)
        self.mix = MDFF(dim)
        self.CPBlocks = nn.ModuleList([IFEBlock(dim) for _ in range(n2)])
        self.patchup = nn.ConvTranspose2d(
            dim, 32, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.BatchNorm2d(32)
        self.gelu = nn.GELU()
        self.final = nn.Conv2d(32, 2, kernel_size=1)
        self.sigma = nn.Parameter(torch.ones(3))
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()
    def forward(self, a, b):
        if self.training:
            a, b = self.randomAB(a, b)
        x1, x2 = a, b
        x1, x2 = self.PE(x1), self.PE(x2)
        x1, x2, diff = self.FES(x1, x2)
        Dis = self.repatch(x1, x2)
        x = self.mix(x1, x2)
        for blk in self.CPBlocks:
            x = blk(x)
        out = self.patchup(x)
        out = self.gelu(self.norm(out))
        out = self.final(out)
        return (out,), (Dis,), (diff,), self.sigma
    def randomAB(self, a, b):
        shape = (a.shape[0], 1, 1, 1)
        random_tensor = torch.rand(shape, dtype=a.dtype, device=a.device)
        random_tensor = random_tensor + 0.5
        random_tensor = random_tensor.floor_()
        return a * random_tensor + b * (1 - random_tensor), b * random_tensor + a * (
            1 - random_tensor
        )
