import torch
import torch.nn as nn

from adaptive_echo_python.encoder import Encoder


class TwoEncoders(nn.Module):
    def __init__(
        self,
        audio_input_size,
        settings_input_size,
        embedding_size,
        hidden_size,
        num_layers,
    ):
        super(TwoEncoders, self).__init__()

        self.audio_encoder = Encoder(
            audio_input_size, embedding_size, hidden_size, num_layers
        )
        self.settings_encoder = Encoder(
            settings_input_size, embedding_size, hidden_size, num_layers
        )

        # hyperparameters for SigLIP loss function
        self.log_t = nn.Parameter(torch.tensor(3.0))
        self.b = nn.Parameter(torch.tensor(-10.0))

    def forward(self, audio_input, settings_input):
        audio_embedding = self.audio_encoder(audio_input)
        settings_embedding = self.settings_encoder(settings_input)

        return audio_embedding, settings_embedding

    @torch.jit.export
    def loss(self, audio_input, settings_input):
        audio_embedding, settings_embedding = self.forward(audio_input, settings_input)

        t = torch.exp(self.log_t)
        z_audio = audio_embedding / torch.norm(audio_embedding, dim=-1, keepdim=True)
        z_settings = settings_embedding / torch.norm(
            settings_embedding, dim=-1, keepdim=True
        )

        # Compute pairwise similarities: (batch_size, embedding_size) x (batch_size, embedding_size) -> (batch_size, batch_size)
        logits = torch.einsum("ae,se->as", z_audio, z_settings) * t + self.b
        batch_size = audio_embedding.shape[0]
        labels = (
            2
            * torch.eye(
                batch_size, device=audio_embedding.device, dtype=audio_embedding.dtype
            )
            - 1
        )

        loss_logits = logits * labels

        # helps with numerical stability
        loss_logits -= loss_logits.max(dim=-1, keepdim=True)[0]
        # log softmax to convert logits to log probabilities
        loss_logits -= torch.log(
            torch.sum(torch.exp(loss_logits), dim=-1, keepdim=True)
        )

        loss = -loss_logits.sum() / batch_size

        return loss

    @torch.jit.export
    def encode_settings(self, settings_input):
        return self.settings_encoder(settings_input)

    @torch.jit.export
    def encode_audio(self, audio_input):
        return self.audio_encoder(audio_input)
