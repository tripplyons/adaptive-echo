from pathlib import Path
from adaptive_echo_python.synth import Synth, synth
from adaptive_echo_python.two_encoders import TwoEncoders
import torch
import numpy as np

num_seconds = 2
training_sample_rate = 8192
num_samples = training_sample_rate * num_seconds
settings_encoder_input_size = Synth().encode_settings().shape[0]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train():
    encoder_embedding_size = 512
    encoder_hidden_size = 512
    encoder_num_layers = 8

    audio_encoder_input_size = TwoEncoders.preprocess_audio(
        torch.randn(1, training_sample_rate * num_seconds)
    ).shape[1]
    print(f"Audio encoder input size: {audio_encoder_input_size}")

    device = get_device()
    print(f"Using device: {device} for training")

    two_encoders = TwoEncoders(
        audio_encoder_input_size,
        settings_encoder_input_size,
        encoder_embedding_size,
        encoder_hidden_size,
        encoder_num_layers,
    ).to(device)

    path = Path("./two_encoders.pt")
    if path.exists():
        print(f"Loading two_encoders from {path}")
        two_encoders = torch.load(path, weights_only=False)
    else:
        print("Creating new two_encoders model")

    two_encoders.train()

    optimizer_muon = torch.optim.Muon(
        [
            {"params": two_encoders.settings_encoder.layers.parameters()},
            {"params": two_encoders.audio_encoder.layers.parameters()},
        ]
    )
    optimizer_adamw = torch.optim.AdamW(
        [
            {"params": two_encoders.settings_encoder.project_in.parameters()},
            {"params": two_encoders.settings_encoder.project_out.parameters()},
            {"params": two_encoders.audio_encoder.project_in.parameters()},
            {"params": two_encoders.audio_encoder.project_out.parameters()},
            {"params": two_encoders.log_t},
            {"params": two_encoders.b},
        ]
    )

    batch_size = 4096
    dataset_size = batch_size * 1
    evaluation_batch_size = 100
    test_dataset_size = 1000

    synth_parallel = torch.compile(
        torch.vmap(synth, in_dims=(0, None), randomness="different")
    )

    def generate_dataset():
        with torch.inference_mode():
            dataset_settings = torch.randn(
                dataset_size, settings_encoder_input_size, device=device
            )
            dataset_audio = []
            while len(dataset_audio) * batch_size < dataset_size:
                new_audio = synth_parallel(
                    dataset_settings[
                        len(dataset_audio) : len(dataset_audio) + batch_size
                    ],
                    torch.linspace(0, num_seconds, num_samples, device=device),
                )
                new_audio = TwoEncoders.preprocess_audio(new_audio)
                dataset_audio.append(new_audio)
                print(
                    f"Generated {len(dataset_audio) * batch_size} of {dataset_size} audio samples"
                )

            dataset_audio = torch.cat(dataset_audio, dim=0)

            return dataset_settings, dataset_audio

    dataset = torch.utils.data.TensorDataset(*generate_dataset())
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    two_encoders = torch.compile(two_encoders)

    print(f"Generating dataset of size {dataset_size}")

    with torch.inference_mode():
        test_settings = torch.randn(
            test_dataset_size, settings_encoder_input_size, device=device
        )
        test_audio = synth_parallel(
            test_settings, torch.linspace(0, num_seconds, num_samples, device=device)
        )
        test_audio = TwoEncoders.preprocess_audio(test_audio)

        test_settings = test_settings.cpu()
        test_audio = test_audio.cpu()

    test_dataset = torch.utils.data.TensorDataset(test_settings, test_audio)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=evaluation_batch_size, shuffle=True
    )

    def evaluate():
        two_encoders.eval()
        with torch.inference_mode():
            losses = []
            settings_accuracies = []
            audio_accuracies = []
            for i, (settings, audio) in enumerate(test_dataloader):
                loss = two_encoders.loss(audio.to(device), settings.to(device))
                losses.append(loss.item())
                audio_accuracy, settings_accuracy = two_encoders.accuracies(
                    audio.to(device), settings.to(device)
                )
                audio_accuracies.append(audio_accuracy.item())
                settings_accuracies.append(settings_accuracy.item())

            return (
                np.mean(losses),
                np.mean(audio_accuracies),
                np.mean(settings_accuracies),
            )

    print("Dataset generated, starting training")

    eval_loss, eval_audio_accuracy, eval_settings_accuracy = evaluate()
    print(
        f"Initial evaluation loss: {eval_loss}, audio accuracy: {eval_audio_accuracy}, settings accuracy: {eval_settings_accuracy}"
    )

    num_epochs = 100

    regenerate_every_n_epochs = 1

    for epoch in range(num_epochs):
        two_encoders.train()
        losses = []

        for settings, audio in dataloader:
            optimizer_muon.zero_grad()
            optimizer_adamw.zero_grad()

            loss = two_encoders.loss(audio, settings)

            loss.backward()
            optimizer_muon.step()
            optimizer_adamw.step()

            losses.append(loss.item())

        print(f"Epoch: {epoch}, average loss: {np.mean(losses)}")

        if (epoch + 1) % regenerate_every_n_epochs == 0:
            dataset = torch.utils.data.TensorDataset(*generate_dataset())
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

        two_encoders.eval()

        with torch.inference_mode():
            eval_loss, eval_audio_accuracy, eval_settings_accuracy = evaluate()
            print(
                f"Epoch: {epoch}, evaluation loss: {eval_loss}, audio accuracy: {eval_audio_accuracy}, settings accuracy: {eval_settings_accuracy}"
            )

            print(f"log_t: {two_encoders.log_t.item()}, b: {two_encoders.b.item()}")

    two_encoders.eval()

    torch.save(two_encoders, path)


if __name__ == "__main__":
    train()
