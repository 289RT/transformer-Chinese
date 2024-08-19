"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn # 从 PyTorch 导入神经网络模块

from models.blocks.encoder_layer import EncoderLayer # 从本地模块导入 EncoderLayer 类
from models.embedding.transformer_embedding import TransformerEmbedding # 从本地模块导入 TransformerEmbedding 类


class Encoder(nn.Module): # 定义一个继承自 nn.Module 的 Encoder 类

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device): # 初始化方法
        super().__init__() # 初始化父类 nn.Module
        self.emb = TransformerEmbedding(d_model=d_model, # 创建 TransformerEmbedding 层的实例
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, # 创建 EncoderLayer 层的列表
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask): # 定义前向传递方法
        x = self.emb(x) # 将输入传递给嵌入层

        for layer in self.layers: # 遍历 EncoderLayer 层
            x = layer(x, src_mask) # 将输出从前一层传递到下一层

        return x # 返回编码器的最终输出
