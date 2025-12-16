import torch
import torch.nn as nn


class selfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(selfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * self.heads == embed_size),"embed_size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim)
        self.fc_out = nn.Linear(self.heads*self.head_dim, self.embed_size)

    def forward(self, values, key, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], key.shape[1], query.shape[1]

