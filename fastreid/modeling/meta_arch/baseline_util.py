import torch
import math
from torch import nn
import torch.nn.functional as F


class PatchMerging(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=dim * 4, out_channels=dim, kernel_size=1, stride=1, padding=0, groups=dim, bias=False
        )
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2
        x = self.conv(x)
        x = self.norm(x)
        return x


class spatical2channel(nn.Module):
    def __init__(self, dim, input_resolution):
        super().__init__()
        self.layers = nn.ModuleList([])
        H, W = input_resolution
        layers= []
        for _ in range(W // 4):
            layers.append(PatchMerging(dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for mergin in self.layers:
            x = mergin(x)
        return x


class channelAttention(nn.Module):
    def __init__(self, dim, input_resolution, dropout):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(dim, dim))
        self.q = spatical2channel(dim, input_resolution)
        self.k = spatical2channel(dim, input_resolution)
        self.scale = dim ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)

    def forward(self, x):
        shortcut = x
        q = self.q(x).squeeze(-1).contiguous()
        k = self.k(x).squeeze(-1).contiguous()

        dots = torch.matmul(q, k.transpose(2, 1)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).squeeze(-1).contiguous()
        support = torch.matmul(self.weights, x)
        out = torch.matmul(attn, support)
        out = out.unsqueeze(-1).transpose(2, 1).reshape(B, C, H, W)
        out = F.relu(out)
        out = out + shortcut
        return out
