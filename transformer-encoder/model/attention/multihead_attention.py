import torch
from .self_attention import Attention
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """

        :param h: 表示有多少个attnetion, 一定能被d_model整除
        :param d_model: 表示bert的hidden_size
        :param dropout: dropout
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_modle = d_model
        self.h_d = d_model // h
        self.linear_transforms = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batchsize = query.shape[0]
        query, key, value = [
            l(x).view(batchsize, -1, self.h, self.h_d).transpose(1,2)# batch_size, head_num, max_len, head_dim
            for l, x in zip(self.linear_transforms, (query, key, value))
        ]
        out, wm = self.attention(query, key, value, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batchsize, -1, self.d_modle)
        out = self.output_linear(out)
        return out


if __name__ == '__main__':
    x = torch.rand((64, 100, 628))
    ml = MultiHeadAttention(4, 628)
    print(ml(x, x, x).shape)
