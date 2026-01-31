import torch
import torch.nn as nn
try:
    from schedulefree import RAdamScheduleFree
except ImportError:
    RAdamScheduleFree = None
from torch.nn import functional as F

from adaptive_echo_python.encoder import Encoder
from adaptive_echo_python.synth import inverse_sigmoid, synth_parallel


class TwoEncoders(nn.Module):
    def __init__(
        self,
        audio_input_size,
        settings_input_size,
        reduced_audio_size,
        embedding_size,
        hidden_size,
        num_layers,
    ):
        super(TwoEncoders, self).__init__()

        self.audio_input_size = audio_input_size
        self.settings_input_size = settings_input_size

        self.reduced_audio_size = reduced_audio_size

        self.n_fft = 4096
        self.hop_length = self.n_fft // 2
        self.window = nn.Parameter(torch.hann_window(self.n_fft), requires_grad=False)
        with torch.inference_mode():
            self.spectrogram_size = torch.stft(
                torch.randn(1, self.audio_input_size, device=torch.device("cpu")),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.to(torch.device("cpu")),
                return_complex=True,
                normalized=True,
                onesided=True,
            ).numel()

        self.audio_reduction_encoder = nn.Sequential(
            nn.Linear(self.spectrogram_size, self.reduced_audio_size),
            nn.GELU(),
            nn.Linear(self.reduced_audio_size, self.reduced_audio_size),
            nn.LayerNorm(self.reduced_audio_size),
        )
        self.audio_reduction_decoder = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.reduced_audio_size, self.spectrogram_size),
        )

        self.audio_encoder = Encoder(
            reduced_audio_size, embedding_size, hidden_size, num_layers
        )
        self.settings_encoder = Encoder(
            settings_input_size, embedding_size, hidden_size, num_layers
        )

        self.audio_embedding_to_settings = nn.Linear(
            embedding_size, settings_input_size
        )

        self.log_t = nn.Parameter(torch.tensor(3.5))
        self.b = nn.Parameter(torch.tensor(-14.0))

    def get_spectrogram(self, audio_input):
        with torch.no_grad():
            audio_input -= torch.mean(audio_input, dim=-1, keepdim=True)
            audio_input /= torch.std(audio_input, dim=-1, keepdim=True)
            spectrogram = torch.stft(
                audio_input,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.to(audio_input.device),
                pad_mode="reflect",
                return_complex=True,
                normalized=True,
                onesided=True,
            )
            spectrogram = torch.abs(spectrogram)
            log_spectrogram = torch.log(spectrogram + 1e-6)

            return log_spectrogram.flatten(start_dim=1)

    def preprocess_audio(self, audio_input):
        spectrogram = self.get_spectrogram(audio_input)
        reduced_audio = self.audio_reduction_encoder(spectrogram)
        return reduced_audio

    def normalize_vector(self, vector):
        vector = vector - torch.mean(vector, dim=-1, keepdim=True)
        vector = vector / torch.norm(vector, dim=-1, keepdim=True)
        return vector

    def fit_and_preprocess_audio(
        self, audio_input, device, batch_size=256, num_epochs=3
    ):
        print(f"audio_input.shape: {audio_input.shape}")
        with torch.inference_mode():
            spectrogram = []
            for i in range(0, len(audio_input), batch_size):
                new_spectrogram = self.get_spectrogram(
                    audio_input[i : i + batch_size].to(device)
                )
                spectrogram.append(new_spectrogram.cpu())
            spectrogram = torch.cat(spectrogram, dim=0)
        print(f"spectrogram.shape: {spectrogram.shape}")

        dataset = torch.utils.data.TensorDataset(spectrogram)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = RAdamScheduleFree(
            [
                {"params": self.audio_reduction_encoder.parameters()},
                {"params": self.audio_reduction_decoder.parameters()},
            ],
            lr=1e-3,
        )
        optimizer.train()
        for i in range(num_epochs):
            losses = []
            for (current_spectrogram,) in dataloader:
                optimizer.zero_grad()
                current_spectrogram = current_spectrogram.to(device)
                reduced_audio = self.audio_reduction_encoder(current_spectrogram)
                reconstructed_spectrogram = self.audio_reduction_decoder(reduced_audio)
                loss = F.mse_loss(reconstructed_spectrogram, current_spectrogram)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(
                f"audio reduction - epoch: {i}, mean loss: {sum(losses) / len(losses)}"
            )
        with torch.inference_mode():
            reduced_audio = []
            for i in range(0, len(spectrogram), batch_size):
                new_reduced_audio = self.audio_reduction_encoder(
                    spectrogram[i : i + batch_size].to(device)
                )
                reduced_audio.append(new_reduced_audio.cpu())
            reduced_audio = torch.cat(reduced_audio, dim=0)
        return reduced_audio

    def forward(self, audio_input, settings_input):
        audio_embedding = self.audio_encoder(
            audio_input,
        )
        settings_embedding = self.settings_encoder(settings_input)

        return audio_embedding, settings_embedding

    @torch.jit.export
    def loss(self, audio_input, settings_input):
        audio_embedding = self.audio_encoder(audio_input)
        settings_embedding = self.settings_encoder(settings_input)

        t = torch.exp(self.log_t)
        normalized_audio = self.normalize_vector(audio_embedding)
        normalized_settings = self.normalize_vector(settings_embedding)

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

        siglip_loss = -loss_logits.sum() / batch_size

        decoded_settings = self.audio_embedding_to_settings(normalized_audio)

        decoder_loss = F.mse_loss(decoded_settings, settings_input)

        loss = siglip_loss + decoder_loss

        return loss

    def accuracies(self, audio_input, settings_input):
        batch_size = audio_input.shape[0]

        audio_embedding = self.audio_encoder(audio_input)
        settings_embedding = self.settings_encoder(settings_input)

        normalized_audio = self.normalize_vector(audio_embedding)
        normalized_settings = self.normalize_vector(settings_embedding)

        logits = torch.einsum("ae,se->as", normalized_audio, normalized_settings)

        audio_predictions = torch.argmax(logits, dim=1)
        audio_labels = torch.arange(batch_size, device=audio_input.device)
        audio_accuracy = (audio_predictions == audio_labels).float().mean()

        settings_predictions = torch.argmax(logits, dim=0)
        settings_labels = torch.arange(batch_size, device=settings_input.device)
        settings_accuracy = (settings_predictions == settings_labels).float().mean()

        return audio_accuracy, settings_accuracy

    def reconstruct_settings(self, audio_input, time):
        self.eval()

        with torch.no_grad():
            settings = self.predict_settings(audio_input)
            settings = inverse_sigmoid(settings)
            print(torch.sigmoid(settings).detach().cpu().numpy())
        settings.requires_grad = True

        with torch.no_grad():
            audio_embedding = self.audio_encoder(self.preprocess_audio(audio_input))
            audio_embedding = self.normalize_vector(audio_embedding)

        optimizer = RAdamScheduleFree(
            [settings],
            weight_decay=0.0,
            lr=3e-3,
        )
        optimizer.train()
        for i in range(2000):
            optimizer.zero_grad()

            generated_audio = synth_parallel(torch.sigmoid(settings), time)

            new_audio_embedding = self.audio_encoder(
                self.preprocess_audio(generated_audio)
            )
            new_audio_embedding = self.normalize_vector(new_audio_embedding)

            new_settings_embedding = self.settings_encoder(settings)
            new_settings_embedding = self.normalize_vector(new_settings_embedding)

            # maximize sum of cosine similarities
            audio_loss = -torch.einsum("ae,ne->", audio_embedding, new_audio_embedding)
            settings_loss = -torch.einsum(
                "ae,se->", audio_embedding, new_settings_embedding
            )
            loss = 0.25 * audio_loss + 0.75 * settings_loss
            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                print(f"loss: {loss.item()}")
                print(torch.sigmoid(settings).detach().cpu().numpy())

        return settings.detach()

    def predict_settings(self, audio_input):
        audio_embedding = self.audio_encoder(self.preprocess_audio(audio_input))
        normalized_audio = self.normalize_vector(audio_embedding)

        return self.audio_embedding_to_settings(normalized_audio)
