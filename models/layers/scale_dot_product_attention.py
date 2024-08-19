"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math  # 导入数学模块以使用平方根函数

from torch import nn  # 从PyTorch库导入神经网络模块


class ScaleDotProductAttention(nn.Module):
    """
    计算缩放点积注意力

    查询：我们关注的给定句子（解码器）
    键：每个句子都要检查与查询的关系（编码器）
    值：每个句子都与键相同（编码器）
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()  # 初始化父类nn.Module
        self.softmax = nn.Softmax(dim=-1)  # 初始化softmax函数，在最后一个维度上操作

    def forward(self, q, k, v, mask=None, e=1e-12):
        # 输入是4维张量
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()  # 获取键张量的维度

        # 1. 点积查询与键^T以计算相似度
        k_t = k.transpose(2, 3)  # 转置键张量
        score = (q @ k_t) / math.sqrt(d_tensor)  # 计算缩放点积

        # 2. 应用掩码（可选）
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)  # 将掩码为0的位置填充为-10000

        # 3. 将它们传递给softmax以使其范围在[0, 1]之间
        score = self.softmax(score)  # 应用softmax函数

        # 4. 乘以值
        v = score @ v  # 计算注意力加权值

        return v, score  # 返回注意力加权值和注意力分数

        # 4. multiply with Value
        v = score @ v

        return v, score
