"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch # 导入 PyTorch 库
from torch import nn # 从 PyTorch 库导入神经网络模块

# Gemini
# LayerNorm 层在神经网络中通常用于稳定和加速训练过程。
# 它通过对每个样本的输入特征进行归一化，使得特征具有零均值和单位方差，
# 从而减少了内部协变量偏移，并且使得不同层之间的特征分布更加一致。这有助于提高模型的泛化能力和收敛速度

class LayerNorm(nn.Module): # 定义一个名为 LayerNorm 的类，继承自 nn.Module
    def __init__(self, d_model, eps=1e-12): # LayerNorm 类的构造函数
        super(LayerNorm, self).__init__() # 初始化父类 nn.Module
        self.gamma = nn.Parameter(torch.ones(d_model)) # 定义可学习参数 gamma，初始化为全 1
        self.beta = nn.Parameter(torch.zeros(d_model)) # 定义可学习参数 beta，初始化为全 0
        self.eps = eps # 定义一个小的常数 eps，防止分母为 0

    def forward(self, x): # LayerNorm 类的前向传播函数
        mean = x.mean(-1, keepdim=True) # 计算输入 x 在最后一个维度上的均值，并保持维度
        var = x.var(-1, unbiased=False, keepdim=True) # 计算输入 x 在最后一个维度上的方差，并保持维度
        # '-1' 表示最后一个维度。

        out = (x - mean) / torch.sqrt(var + self.eps) # 对输入 x 进行层归一化
        out = self.gamma * out + self.beta # 对层归一化后的结果进行缩放和平移
        return out # 返回层归一化后的结果
