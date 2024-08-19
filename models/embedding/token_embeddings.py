"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    使用 torch.nn 进行词元嵌入
    它们将使用加权矩阵对单词进行密集表示
    """

    def __init__(self, vocab_size, d_model):
        """
        包含位置信息的词元嵌入类

        :param vocab_size: 词汇表大小
        :param d_model: 模型维度
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
