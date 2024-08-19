"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention

"""多头注意力机制的核心思想是并行计算多个注意力表示，从而捕捉到输入序列中不同方面的依赖关系。

具体来说，多头注意力机制将输入序列分别通过多个独立的注意力头进行处理。
每个注意力头都有一组独立的参数，可以学习到不同的注意力表示。这些注意力表示最后被拼接起来，并通过一个线性变换得到最终的输出。

这样做的好处主要有以下几点：

捕捉更丰富的上下文信息：不同的注意力头可以关注输入序列的不同部分，从而捕捉到更全面的上下文信息。
提高模型的表达能力：通过并行计算多个注意力表示，模型可以学习到更复杂的特征表示。
增强模型的鲁棒性：多个注意力头可以相互补充，减少单个注意力头出错带来的影响。
"""

class MultiHeadAttention(nn.Module):
    # 多头注意力机制的实现

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head  # 注意力头的数量
        self.attention = ScaleDotProductAttention()  # 缩放点积注意力层
        self.w_q = nn.Linear(d_model, d_model)  # 查询向量的线性变换
        self.w_k = nn.Linear(d_model, d_model)  # 键向量的线性变换
        self.w_v = nn.Linear(d_model, d_model)  # 值向量的线性变换
        self.w_concat = nn.Linear(d_model, d_model)  # 拼接后的向量的线性变换

    def forward(self, q, k, v, mask=None):
        # 前向传播
        # 1. 使用权重矩阵进行点积
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)  # 对查询、键、值向量进行线性变换

        # 2. 按注意力头的数量分割张量
        q, k, v = self.split(q), self.split(k), self.split(v)  # 将张量分割成多个头

        # 3. 进行缩放点积以计算相似度
        out, attention = self.attention(q, k, v, mask=mask)  # 计算注意力权重和输出

        # 4. 拼接并传递给线性层
        out = self.concat(out)  # 将多个头的输出拼接起来
        out = self.w_concat(out)  # 对拼接后的向量进行线性变换

        # 5. 可视化注意力图
        # TODO : 我们应该实现可视化功能

        return out  # 返回多头注意力机制的输出

    def split(self, tensor):
        # 按注意力头的数量分割张量
        """
        按注意力头的数量分割张量

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()  # 获取张量的维度

        d_tensor = d_model // self.n_head  # 计算每个头的维度
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # 类似于分组卷积（按注意力头的数量分割）

        return tensor  # 返回分割后的张量

    def concat(self, tensor):
        # self.split(tensor : torch.Tensor) 的逆函数
        """
        self.split(tensor : torch.Tensor) 的逆函数

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()  # 获取张量的维度
        d_model = head * d_tensor  # 计算原始张量的维度

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        # 将多个头的输出拼接起来
        return tensor  # 返回拼接后的张量
