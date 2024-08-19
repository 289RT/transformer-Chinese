"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch # 导入 PyTorch 库，用于构建神经网络。
from torch import nn # 导入 PyTorch 的神经网络模块。

from models.model.decoder import Decoder # 从 models.model.decoder 模块导入 Decoder 类。
from models.model.encoder import Encoder # 从 models.model.encoder 模块导入 Encoder 类。


class Transformer(nn.Module): # 定义一个 Transformer 类，继承自 nn.Module。

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device): # Transformer 类的构造函数。
        super().__init__() # 初始化父类 nn.Module。
        self.src_pad_idx = src_pad_idx # 源语言填充索引。
        self.trg_pad_idx = trg_pad_idx # 目标语言填充索引。
        self.trg_sos_idx = trg_sos_idx # 目标语言开始符号索引。
        self.device = device # 设备（CPU 或 GPU）。
        self.encoder = Encoder(d_model=d_model, # 创建 Encoder 对象。
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model, # 创建 Decoder 对象。
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg): # Transformer 类的前向传播方法。
        src_mask = self.make_src_mask(src) # 创建源语言掩码。
        trg_mask = self.make_trg_mask(trg) # 创建目标语言掩码。
        enc_src = self.encoder(src, src_mask) # 对源语言进行编码。
        output = self.decoder(trg, enc_src, trg_mask, src_mask) # 对目标语言进行解码。
        return output # 返回解码结果。

    def make_src_mask(self, src): # 创建源语言掩码的方法。
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # 计算掩码。
        return src_mask # 返回掩码。

    def make_trg_mask(self, trg): # 创建目标语言掩码的方法。
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3) # 计算填充掩码。
        trg_len = trg.shape[1] # 目标语言序列长度。
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device) # 计算子序列掩码。
        trg_mask = trg_pad_mask & trg_sub_mask # 结合填充掩码和子序列掩码。
        return trg_mask # 返回掩码。
