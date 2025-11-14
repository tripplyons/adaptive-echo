import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(EncoderLayer, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()

        self.linear.weight.data.normal_(0, 1e-3)
        self.linear.bias.data.zero_()

    def forward(self, x):
        residual = self.linear(x)
        residual = self.gelu(residual)
        residual = self.dropout(residual) 

        return self.norm(x + residual)


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
