from .embedding import TransformerEmbedding
from .block import TransformerBlock
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, d_model, d_ff, dropout, vocab, max_len, h, N):
        """

        :param d_model: hidden_size
        :param d_ff: position-wise FFn的隐藏层的维度
        :param dropout: dropout
        :param vocab: 词典大小
        :param max_len: 句子长度
        :param h: 多头注意力机制的头数
        :param N: transformerblock的个数
        """
        super(Transformer, self).__init__()
        self.emb = TransformerEmbedding(vocab, d_model, max_len, dropout)
        self.block = TransformerBlock(d_model, d_ff, h, dropout)
        self.blocks = nn.ModuleList([self.block for _ in range(N)])

    def forward(self, batch):
        input = self.emb(batch)
        for i, block in enumerate(self.block):
            out = block(input)

        return out