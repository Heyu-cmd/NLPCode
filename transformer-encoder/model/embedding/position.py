import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False
        pos = torch.arange(0, max_len).unsqueeze(1)
        term = torch.exp(-math.log(10000) / d_model * torch.arange(0, d_model, 2))
        pe[:, 0::2] = torch.sin(term * pos)
        pe[:, 1::2] = torch.cos(term * pos)

        pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        length = x.size(0)
        return self.pe[:length, :]


if __name__ == '__main__':
    ps = PositionEmbedding(d_model=628)
    print(ps(torch.arange(0, 100)).shape)
