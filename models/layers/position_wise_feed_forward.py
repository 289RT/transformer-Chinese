"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn  # 导入 PyTorch 的神经网络模块


class PositionwiseFeedForward(nn.Module):  # 定义一个名为 PositionwiseFeedForward 的类，继承自 nn.Module

    def __init__(self, d_model, hidden, drop_prob=0.1):  # 初始化函数，定义模型的层
        super(PositionwiseFeedForward, self).__init__()  # 初始化父类 nn.Module
        self.linear1 = nn.Linear(d_model, hidden)  # 第一个线性层，将输入维度 d_model 映射到 hidden 维度
        self.linear2 = nn.Linear(hidden, d_model)  # 第二个线性层，将 hidden 维度映射回 d_model 维度
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.dropout = nn.Dropout(p=drop_prob)  # Dropout 层，防止过拟合
        
# Dropout层是一种正则化技术，用于减少神经网络中的过拟合
# 在训练过程中，Dropout层会随机“丢弃”一部分神经元（及其连接），从而防止神经元之间过度依赖，迫使网络学习更鲁棒的特征表示
    
    def forward(self, x):  # 定义前向传播函数
        x = self.linear1(x)  # 通过第一个线性层
        x = self.relu(x)  # 应用 ReLU 激活函数
        x = self.dropout(x)  # 应用 Dropout
        x = self.linear2(x)  # 通过第二个线性层
        return x  # 返回输出
