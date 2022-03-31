from .attention import MultiHeadAttention
from .utils import LayerNorm, PositionWiseFeedForward
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout):
        super(TransformerBlock, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.mlatt = MultiHeadAttention(h, d_model, dropout)

    def forward(self, x):
        l1 = self.mlatt(x, x, x)
        l1 = self.layer_norm(x + l1)
        l2 = self.ffn(l1)
        l2 = self.layer_norm(l1 + l2)
        return l2


if __name__ == '__main__':
    import torch

    block = TransformerBlock(768, 1024, 4, 0.2)
    x = torch.rand(64, 100, 768)
    y = block(x)
    print(y.shape)
