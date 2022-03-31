import torch.nn as nn
import torch


class LayerNorm(nn.Module):

    def __init__(self, size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        # a2 和 b2 表示缩放和平移的参数向量
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        res = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return res


if __name__ == '__main__':
    ln = LayerNorm(768)
    s = torch.rand(20, 100, 768)
    print(ln(s).shape)