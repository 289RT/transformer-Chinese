"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    计算正弦编码。
    """

    def __init__(self, d_model, max_len, device):
        """
        正弦编码类的构造函数

        :param d_model: 模型的维度
        :param max_len: 最大序列长度
        :param device: 硬件设备设置
        """
        super(PositionalEncoding, self).__init__()

        # 与输入矩阵大小相同（用于与输入矩阵相加）
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 我们不需要计算梯度

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze 表示单词的位置

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' 表示 d_model 的索引（例如，嵌入大小 = 50，'i' = [0,50]）
        # "step=2" 表示 'i' 乘以 2（与 2 * i 相同）

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # 计算位置编码以考虑单词的位置信息

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # 它将与 tok_emb 相加：[128, 30, 512]
