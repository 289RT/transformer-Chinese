"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        # 初始化解码器模块的父类
        super().__init__() 
        # 初始化词嵌入层
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)
        # 初始化解码器层列表
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        # 初始化线性层，将解码器输出映射到目标词汇表大小
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # 对目标序列进行词嵌入
        trg = self.emb(trg)
        # 通过每一层解码器层传递嵌入的目标序列
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        # 将最终解码器层的输出传递给线性层
        output = self.linear(trg)
        # 返回线性层的输出
        return output
