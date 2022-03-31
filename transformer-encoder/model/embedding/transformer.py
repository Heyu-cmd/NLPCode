from .position import PositionEmbedding
from .token import TokenEmbedding
import torch.nn as nn

class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.te = TokenEmbedding(vocab_size, d_model)
        self.pe = PositionEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    def forward(self, x):
        out = self.te(x) + self.pe(x)

        return out


