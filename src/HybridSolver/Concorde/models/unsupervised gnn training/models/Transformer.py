from torch import nn
import torch
from torch.nn import functional as F
from torch_geometric.nn import MLP

class SelfAttention(nn.MultiheadAttention):
    def forward(self, x, *args, **kwargs):
        return super().forward(
            query=x, key=x, value=x, need_weights=False, *args, **kwargs)[0]


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.normalizer = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, input):
        return self.normalizer(input.swapaxes(1, 2)).swapaxes(1, 2)


class TransformerLayer(nn.Sequential):
    def __init__(self, num_attention_heads, embedding_dim, feed_forward_dim, activation=nn.ReLU()):
        super().__init__(
            SkipConnection(
                SelfAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    batch_first=True
                )
            ),
            Normalization(embedding_dim=embedding_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(in_features=embedding_dim, out_features=feed_forward_dim),
                    activation,
                    nn.Linear(in_features=feed_forward_dim, out_features=embedding_dim)
                )
            ),
            Normalization(embedding_dim=embedding_dim),
        )

class Transformer(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim, num_attention_heads, num_attention_layers, activation=nn.ReLU()):
        super().__init__()
        self.proj = nn.Linear(2, embedding_dim)
        self.layers = nn.Sequential(*(
            TransformerLayer(num_attention_heads, embedding_dim, feed_forward_dim, activation)
            for _ in range(num_attention_layers)
        ))
        
    def forward(self, x):
        
        x = self.proj(x)
        x = self.layers(x)
        
        return x