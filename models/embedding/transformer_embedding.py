"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn # 从 PyTorch 导入神经网络模块。

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding 
# 从你的模型定义中导入位置编码和标记嵌入模块。

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """
    # 定义一个名为 TransformerEmbedding 的类，它继承自 nn.Module，表示一个 PyTorch 神经网络模块。
    # 它将标记嵌入和位置编码结合起来，为网络提供位置信息。

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        # TransformerEmbedding 类的构造函数，它初始化类的属性。
        # 它接受词汇量大小、模型维度、最大序列长度、dropout 概率和设备作为输入。

        super(TransformerEmbedding, self).__init__() 
        # 初始化父类 nn.Module，确保正确设置所有必要的属性和方法。

        self.tok_emb = TokenEmbedding(vocab_size, d_model) 
        # 创建一个 TokenEmbedding 层，将标记转换为密集向量表示。

        self.pos_emb = PositionalEncoding(d_model, max_len, device) 
        # 创建一个 PositionalEncoding 层，将位置信息添加到标记嵌入中。

        self.drop_out = nn.Dropout(p=drop_prob) 
        # 创建一个 dropout 层，以防止过拟合，通过在训练期间随机丢弃一些神经元。

    def forward(self, x):
        # 定义 TransformerEmbedding 模块的前向传递。
        # 它接受输入张量 x 并通过标记嵌入、位置编码和 dropout 层进行传递。

        tok_emb = self.tok_emb(x) 
        # 通过 TokenEmbedding 层传递输入，获得标记嵌入。

        pos_emb = self.pos_emb(x) 
        # 通过 PositionalEncoding 层传递输入，获得位置编码。

        return self.drop_out(tok_emb + pos_emb) 
        # 将标记嵌入和位置编码相加，应用 dropout，并返回结果。
