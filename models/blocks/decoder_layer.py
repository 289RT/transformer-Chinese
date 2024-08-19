"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""

from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module): # 定义一个解码器层类，继承自nn.Module

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob): # 初始化函数，定义模型参数
        super(DecoderLayer, self).__init__() # 初始化父类
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head) # 定义自注意力机制
        self.norm1 = LayerNorm(d_model=d_model) # 定义层归一化
        self.dropout1 = nn.Dropout(p=drop_prob) # 定义dropout层

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head) # 定义编码器-解码器注意力机制
        self.norm2 = LayerNorm(d_model=d_model) # 定义层归一化
        self.dropout2 = nn.Dropout(p=drop_prob) # 定义dropout层

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob) # 定义前馈神经网络
        self.norm3 = LayerNorm(d_model=d_model) # 定义层归一化
        self.dropout3 = nn.Dropout(p=drop_prob) # 定义dropout层

    def forward(self, dec, enc, trg_mask, src_mask): # 定义前向传播函数
        # 1. compute self attention # 计算自注意力
        _x = dec # 保存输入
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask) # 计算自注意力
        
        # 2. add and norm # 残差连接和层归一化
        x = self.dropout1(x) # dropout
        x = self.norm1(x + _x) # 层归一化

        if enc is not None: # 如果有编码器输入
            # 3. compute encoder - decoder attention # 计算编码器-解码器注意力
            _x = x # 保存输入
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask) # 计算编码器-解码器注意力
            
            # 4. add and norm # 残差连接和层归一化
            x = self.dropout2(x) # dropout
            x = self.norm2(x + _x) # 层归一化

        # 5. positionwise feed forward network # 前馈神经网络
        _x = x # 保存输入
        x = self.ffn(x) # 前馈神经网络
        
        # 6. add and norm # 残差连接和层归一化
        x = self.dropout3(x) # dropout
        x = self.norm3(x + _x) # 层归一化
        return x # 返回输出
