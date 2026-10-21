import torch.nn as nn
import torch.nn.functional as F
import torch
from kan import KAN  # 假设 KAN 是一个自定义的模块，包含 KAN 类
from src.model import StockMixer 
from kan import KAN


stock_num = 2000
lookback_length = 1
fea_num = 1400
market_num = 128
scale_factor = 3
model = StockMixer(
    factor_num=fea_num
)


m = stock_num
n = fea_num
x_input = torch.randn(m, n) 
output = model(x_input)

print("输入形状:", x_input.shape)
print("输出形状:", output.shape)

# m = 1000
# n = fea_num
# x_input = torch.randn(m, n) 
# output = model(x_input)

# print("输入形状:", x_input.shape)
# print("输出形状:", output.shape)


