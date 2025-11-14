import torch
from torch.nn import functional as F
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
        conv_hidden_size = 3

        self.audio_sequential = nn.Sequential(
            nn.Conv1d(1, conv_hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_hidden_size, 1, kernel_size=3, padding=1),
        )

        self.audio_encoder = Encoder(
            audio_input_size, embedding_size, hidden_size, num_layers
        )
        self.settings_encoder = Encoder(
            settings_input_size, embedding_size, hidden_size, num_layers
        )

        # learnable hyperparameters for SigLIP loss function
        self.log_t = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def preprocess_audio(audio_input):
        with torch.no_grad():
            audio_input = audio_input / torch.norm(audio_input, dim=-1, keepdim=True)
            spectrogram = torch.stft(
                audio_input,
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                window=torch.hann_window(4096).to(audio_input.device),
                pad_mode="reflect",
                return_complex=True,
                normalized=True,
                onesided=True,
            )
            spectrogram = torch.abs(spectrogram)

            return spectrogram.flatten(start_dim=1)

    def forward(self, audio_input, settings_input):
        audio_embedding = self.encode_audio(
            audio_input,
        )
        settings_embedding = self.encode_settings(settings_input)

        return audio_embedding, settings_embedding

    @torch.jit.export
    def loss(self, audio_embedding, settings_embedding):
        return self.siglip_loss(audio_embedding, settings_embedding)

    @torch.jit.export
    def siglip_loss(self, audio_input, settings_input):
        audio_embedding, settings_embedding = self.forward(audio_input, settings_input)

        t = torch.exp(self.log_t)
        normalized_audio = audio_embedding / torch.norm(
            audio_embedding, dim=-1, keepdim=True
        )
        normalized_settings = settings_embedding / torch.norm(
            settings_embedding, dim=-1, keepdim=True
        )
        normalized_audio = audio_embedding
        normalized_settings = settings_embedding

        # Compute pairwise similarities: (batch_size, embedding_size) x (batch_size, embedding_size) -> (batch_size, batch_size)
        logits = (
            torch.einsum("ae,se->as", normalized_audio, normalized_settings) * t
            + self.b
        )
        batch_size = audio_embedding.shape[0]
        labels = (
            2
            * torch.eye(
                batch_size, device=audio_embedding.device, dtype=audio_embedding.dtype
            )
            - 1
        )

        loss_logits = F.logsigmoid(logits * labels)
        loss = -loss_logits.sum() / batch_size

        return loss

    @torch.jit.export
    def encode_settings(self, settings_input):
        return self.settings_encoder(settings_input)

    @torch.jit.export
    def encode_audio(self, audio_input):
        audio_sequential_result = self.audio_sequential(audio_input.unsqueeze(-2))[
            ..., 0, :
        ]
        return self.audio_encoder(audio_sequential_result)

    def accuracies(self, audio_input, settings_input):
        batch_size = audio_input.shape[0]

        audio_embedding, settings_embedding = self.forward(audio_input, settings_input)

        normalized_audio = audio_embedding / torch.norm(
            audio_embedding, dim=-1, keepdim=True
        )
        normalized_settings = settings_embedding / torch.norm(
            settings_embedding, dim=-1, keepdim=True
        )

        logits = torch.einsum("ae,se->as", normalized_audio, normalized_settings)

        audio_predictions = torch.argmax(logits, dim=1)
        audio_labels = torch.arange(batch_size, device=audio_input.device)
        audio_accuracy = (audio_predictions == audio_labels).float().mean()

        settings_predictions = torch.argmax(logits, dim=0)
        settings_labels = torch.arange(batch_size, device=settings_input.device)
        settings_accuracy = (settings_predictions == settings_labels).float().mean()

        return audio_accuracy, settings_accuracy
