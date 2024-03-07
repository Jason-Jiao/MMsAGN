import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .gcn_dhg_util import adj2adj, normalize_features, adj2dhg, generate_G_from_H


class GCNLayer(nn.Module):
    def __init__(self, input_features, output_features, num_heads, bias=False):
        super(GCNLayer, self).__init__()
        self.input_features = input_features  # 输入特征维度
        self.output_features = output_features  # 输出特征维度
        self.weights = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.num_heads = num_heads
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj_attn, x):
        support = torch.matmul(x, self.weights)
        support = rearrange(support, "b n (h d) -> b h n d", h=self.num_heads)

        output = torch.matmul(adj_attn, support)
        output = rearrange(output, 'b h n d -> b n (h d)')
        if self.bias is not None:
            return output + self.bias
        return output


class DHGLayer(nn.Module):
    def __init__(self, input_features, output_features, num_heads, bias=False):
        super(DHGLayer, self).__init__()
        self.input_features = input_features  # 输入特征维度
        self.output_features = output_features  # 输出特征维度
        self.weights = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.W_weights = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.W = nn.Parameter(torch.FloatTensor(input_features // num_heads, 1))
        # self.to_W = nn.Linear(256, 256, bias=False)  # 超图W 256为节点个数
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        std = 1. / math.sqrt(self.W_weights.size(1))
        self.W_weights.data.uniform_(-std, std)
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj_attn, x):
        support = torch.matmul(x, self.weights)
        # print(support)
        # support = x

        W_support = torch.matmul(x, self.W_weights)
        support = rearrange(support, "b n (h d) -> b h n d", h=self.num_heads)
        W_support = rearrange(W_support, "b n (h d) -> b h n d", h=self.num_heads)
        W_support = torch.matmul(adj_attn, W_support)
        W = torch.matmul(W_support, self.W).squeeze()
        W = self.softmax(W)
        adj_attn = adj2dhg(adj_attn)

        G = generate_G_from_H(adj_attn, W)
        # G = generate_G_from_H(adj_attn)
        output = torch.matmul(G, support)
        # print("output", torch.isnan(output).any().item())
        # print("output_inf", torch.isinf(output).any().item())
        output = rearrange(output, 'b h n d -> b n (h d)')
        if self.bias is not None:
            return output + self.bias
        return output


class imageGCN_DHG(nn.Module):
    def __init__(self, input_features, output_features, num_heads, dropout):
        super(imageGCN_DHG, self).__init__()
        self.dropout = dropout
        # 初始化图卷积层
        self.gcn1 = GCNLayer(input_features, output_features, num_heads, bias=True)
        # self.gcn2 = GCNLayer(output_features, output_features, num_heads, bias=True)
        self.dhg1 = DHGLayer(input_features, output_features, num_heads, bias=True)
        # self.dhg2 = DHGLayer(output_features, output_features, num_heads, bias=True)

    def forward(self, adj_matrix, x):
        x = normalize_features(x)
        # 使用 ReLU 作为激活函数进行图卷积
        x = F.relu(self.dhg1(adj_matrix, x))
        x = F.dropout(x, self.dropout)
        x = self.gcn1(adj_matrix, x)
        return x
