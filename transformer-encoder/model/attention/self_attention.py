import torch.nn.functional as F
import torch.nn as nn
import torch


class Attention(nn.Module):

    def forward(self, query, key, value, dropout=None):
        wm = torch.matmul(query, key.transpose(-2, -1)) / query.shape[-1]
        wm = F.softmax(wm, dim=-1)
        if dropout is not None:
            wm = dropout(wm)
        return torch.matmul(wm, value), wm


if __name__ == '__main__':
    x = torch.rand((64, 100, 628))
    att = Attention()
    out, wm = att(x,x,x)
    print(out.shape)


