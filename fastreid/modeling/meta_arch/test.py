import numpy as np
import torch
# from .baseline_util import SELayer

# from baseline_util import channelAttention


# model = channelAttention(2048,(16,16),0.3)
# # features= torch.randn(64,256,2048)
features = torch.randn(64, 2048, 16, 16)
# # print("@#@#@#@",np.percentile(features,25))
# result = model(features)
# print(result.shape)
# a = [1, 2, 3, 4, 5, 6, 6, 7, 5, 4, 3, 323, 2, 2]
# a = torch.tensor(a)
# print(a)
# print("@#@#@#@",features.quantile(0.75))
# features = torch.randn(64, 2048, 16, 16)
# x = torch.randn(64, 2048, 16, 16)
# se = SELayer(channel=2048, reduction=4)
# result = se(x, features)
# print(result.shape)
# import torch
# import torch
# import torch.nn.functional as F

# 创建一个 Tensor
# tensor = torch.tensor([1, 2, 3, 4, 5])

# 计算 Tensor 的中位数
median = torch.quantile(features, 0.5)

# 打印结果
print(median)

