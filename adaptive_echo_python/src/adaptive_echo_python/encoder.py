import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(EncoderLayer, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.project_in = nn.Linear(embedding_size, hidden_size)
        self.project_out = nn.Linear(hidden_size, embedding_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.project_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.project_out(x)
        return x
