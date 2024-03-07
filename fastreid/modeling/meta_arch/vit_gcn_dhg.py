import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import Sequential as Seq
from .image_gcn_dhg import imageGCN_DHG
from .gcn_dhg_util import calculate_quantile, mask_adj_matrix


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 初始化图卷积层
        self.gcn_dhg = imageGCN_DHG(dim, dim, heads, dropout)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        # attn = self.dropout(attn)
        # muti-gcn
        adj_matrix = attn

        # erase
        mask_threshold = calculate_quantile(adj_matrix, 0.95) 
        # mask_threshold = torch.quantile(adj_matrix, q=0.95) # gradcam可视化时使用
        adj_matrix = mask_adj_matrix(adj_matrix, mask_threshold)

        gcn_dhg_out = self.gcn_dhg(adj_matrix, x)
        return self.to_out(gcn_dhg_out)


class Transformer_gcn_dhg(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        # grad-cam,钩子
        self.identity = nn.Identity()

    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).transpose(2, 1).squeeze(-1).contiguous()
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.unsqueeze(-1).transpose(2, 1).reshape(B, C, H, W)
        x = self.identity(x)
        return x
