"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """初始化。

        参数：
            d_model: 模型维度
            ffn_hidden: 前馈网络中隐藏层的维度
            n_head: 多头注意力中的头数
            drop_prob: dropout 概率
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head) # 多头注意力层
        self.norm1 = LayerNorm(d_model=d_model) # 层归一化层
        self.dropout1 = nn.Dropout(p=drop_prob) # dropout 层

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob) # 前馈网络
        self.norm2 = LayerNorm(d_model=d_model) # 层归一化层
        self.dropout2 = nn.Dropout(p=drop_prob) # dropout 层

    def forward(self, x, src_mask):
        """前向传播。

        参数：
            x: 输入张量
            src_mask: 源掩码

        返回：
            输出张量
        """
        # 1. 计算自注意力
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask) # 计算自注意力

        # 2. 添加残差连接并进行层归一化
        x = self.dropout1(x) # 应用 dropout
        x = self.norm1(x + _x) # 添加残差连接并应用层归一化

        # 3. 应用位置前馈网络
        _x = x
        x = self.ffn(x) # 应用位置前馈网络

        # 4. 添加残差连接并进行层归一化
        x = self.dropout2(x) # 应用 dropout
        x = self.norm2(x + _x) # 添加残差连接并应用层归一化
        return x
