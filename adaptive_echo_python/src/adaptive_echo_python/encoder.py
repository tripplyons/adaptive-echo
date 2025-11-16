import torch.nn as nn
from torch.nn import functional as F


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = self.linear(x)
        residual = self.gelu(residual)
        residual = self.dropout(residual)

        x = x + residual
        x = F.layer_norm(x, x.shape[-1:])

        return x


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.project_in = nn.Linear(input_size, hidden_size)
        self.project_out = nn.Linear(hidden_size, embedding_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        x = self.project_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.project_out(x)
        return x
