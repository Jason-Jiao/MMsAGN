import torch
import torch.nn as nn


def normalize_features(features):
    rowsum = torch.abs(features).sum(dim=1)
    features = features / rowsum.unsqueeze(1)
    return features


def adj2adj(adjacency_matrix):
    A_hat = torch.tensor(adjacency_matrix, dtype=torch.float)
    # 计算度矩阵 D
    D_hat = torch.diag(torch.sqrt(1.0 / (A_hat.sum(dim=1) + 1e-6)))

    # 对称归一化
    A_norm = D_hat @ A_hat @ D_hat

    return A_norm


def adj2dhg(adj_matrix):
    # 转置
    adj_matrix = adj_matrix.transpose(-1, -2)
    # 创建与邻接矩阵相同形状的全零张量
    dhg_matrix = torch.zeros_like(adj_matrix)
    # 将大于0的数值置为1
    dhg_matrix[adj_matrix > 0] = 1
    return dhg_matrix


# 计算矩阵的负次方并处理无穷大值(停用)
def matrix_negative_power(matrix, power):
    result = torch.pow(matrix, power)
    result = torch.where(torch.isinf(result), torch.zeros_like(result), result)
    return result


# 擦除操作，对角线元素不擦除
def mask_adj_matrix(adj_matrix, mask_threshold):
    # 创建掩码，将小于等于mask_threshold的元素置为0
    mask = adj_matrix <= mask_threshold

    # 掩码条件：不在对角线上的位置
    diag_mask = torch.eye(adj_matrix.size(-1)).bool().unsqueeze(0).unsqueeze(0).cuda()
    mask = mask & (~diag_mask)

    # 将满足掩码条件的元素置为0
    adj_matrix = adj_matrix.masked_fill(mask, 0)

    return adj_matrix


def generate_G_from_H(H, W):
# def generate_G_from_H(H):
    """
    Calculate G from hypergraph incidence matrix H
    :param H: Hypergraph incidence matrix H
    :param variable_weight: Whether the weight of hyperedge is variable
    :return: G
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = H.to(device)
    # n_batch, n_heads, n_node, n_edge = H.size()
    # W = torch.ones(n_batch, n_heads, n_edge).to(device)

    # The degree of the node
    DV = torch.sum(H * W.unsqueeze(2), dim=3)

    # The degree of the hyperedge
    DE = torch.sum(H, dim=2)

    # invDE = torch.diag_embed(matrix_negative_power(DE, -1))
    # DV2 = torch.diag_embed(matrix_negative_power(DV, -0.5))
    invDE = torch.diag_embed(DE.pow(-1))
    DV2 = torch.diag_embed(DV.pow(-0.5))
    W = torch.diag_embed(W)
    H = H.permute(0, 1, 3, 2)

    # if variable_weight:
    #     DV2_H = torch.matmul(DV2, H)
    #     invDE_HT_DV2 = torch.matmul(torch.matmul(invDE, H.permute(0, 1, 3, 2)), DV2)
    #     return DV2_H, self.W, invDE_HT_DV2
    # else:
    #     G = torch.matmul(torch.matmul(torch.matmul(DV2, H), self.W), torch.matmul(invDE, H.permute(0, 1, 3, 2))).matmul(DV2)
    #     return G
    # DV2_H = torch.matmul(DV2, H)
    # invDE_HT_DV2 = torch.matmul(torch.matmul(invDE, H.permute(0, 1, 3, 2)), DV2)
    # return DV2_H, W, invDE_HT_DV2
    G = torch.matmul(torch.matmul(torch.matmul(DV2, H), W), torch.matmul(invDE, H.permute(0, 1, 3, 2))).matmul(DV2)
    return G


def calculate_quantile(input_tensor, quantile=0.95):
    # 将输入张量分成两半
    half_point = input_tensor.size(0) // 2
    first_half = input_tensor[:half_point]
    second_half = input_tensor[half_point:]

    # 计算每半部分的指定分位数
    quantile_1 = torch.quantile(first_half, q=quantile)
    quantile_2 = torch.quantile(second_half, q=quantile)

    # 将两个结果相加
    result = (quantile_1 + quantile_2) / 2.0

    return result


if __name__ == '__main__':
    # 邻接矩阵是一个二维数组
    adjacency_matrix = [[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]]

    print(adj2adj(adjacency_matrix))
    features = torch.randn(64, 1024, 16, 16)
    features = normalize_features(features)
    print(features)
