from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from adaptive_echo_python.synth import Synth, synth
from adaptive_echo_python.two_encoders import TwoEncoders

num_seconds = 2
training_sample_rate = 8192
num_samples = training_sample_rate * num_seconds
settings_encoder_input_size = Synth().encode_settings().shape[0]
reduced_audio_size = 1024
encoder_embedding_size = 512
encoder_hidden_size = 512
encoder_num_layers = 6
learning_rate = 1e-4
gradient_clip_value = 1.0
weight_decay = 0.0
batch_size = 8192
dataset_size = batch_size * 10
evaluation_batch_size = 1000
test_dataset_size = evaluation_batch_size * 5
num_epochs = 1000000 # basically forever
regenerate_dataset_every_n_epochs = 10
evaluate_every_n_epochs = 10
save_every_n_epochs = 10
generate_sound_every_n_epochs = 10
model_path = Path("./two_encoders.pt")
optimizer_path = Path("./two_encoders_optimizer.pt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train():
    audio_encoder_input_size = num_samples

    device = get_device()
    print(f"Using device: {device} for training")

    two_encoders = TwoEncoders(
        audio_encoder_input_size,
        settings_encoder_input_size,
        reduced_audio_size,
        encoder_embedding_size,
        encoder_hidden_size,
        encoder_num_layers,
    )

    existing_model = model_path.exists()
    if existing_model:
        print(f"Loading two_encoders from {model_path}")
        two_encoders = torch.load(model_path, weights_only=False)
    else:
        print("Creating new two_encoders model")

    two_encoders.to(device)

    foreach = device.type == "cuda"

    optimizer = torch.optim.AdamW(
        two_encoders.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.95, 0.9995),
        foreach=foreach,
    )

    two_encoders.train()

    synth_parallel = torch.compile(
        torch.vmap(synth, in_dims=(0, None), randomness="different")
    )

    def generate_dataset(
        dataset_size, dataset_device, synth_device, preprocess=True, verbose=False
    ):
        with torch.inference_mode():
            dataset_audio = []
            dataset_settings = []
            while len(dataset_audio) * batch_size < dataset_size:
                new_settings = torch.randn(
                    batch_size, settings_encoder_input_size, device=synth_device
                )
                new_audio = synth_parallel(
                    new_settings,
                    torch.linspace(0, num_seconds, num_samples, device=synth_device),
                )
                if preprocess:
                    new_audio = two_encoders.preprocess_audio(new_audio)
                dataset_audio.append(new_audio.to(dataset_device, non_blocking=True))
                dataset_settings.append(
                    new_settings.to(dataset_device, non_blocking=True)
                )
                if verbose:
                    print(
                        f"Generated {len(dataset_audio) * batch_size} of {dataset_size} audio samples"
                    )

            # block until all data is transferred
            if device.type == "cuda":
                torch.cuda.synchronize()
            if device.type == "mps":
                torch.mps.synchronize()

            dataset_audio = torch.cat(dataset_audio, dim=0)
            dataset_settings = torch.cat(dataset_settings, dim=0)

            return dataset_settings, dataset_audio

    dataset_settings, dataset_audio = generate_dataset(
        dataset_size, torch.device("cpu"), device, preprocess=False
    )

    two_encoders = torch.compile(two_encoders)

    if existing_model:
        with torch.inference_mode():
            dataset_reduced_audio = []
            for i in range(0, len(dataset_audio), batch_size):
                dataset_reduced_audio.append(
                    two_encoders.preprocess_audio(
                        dataset_audio[i : i + batch_size].to(device)
                    ).cpu()
                )
            dataset_reduced_audio = torch.cat(dataset_reduced_audio, dim=0)
    else:
        two_encoders.train()
        dataset_reduced_audio = two_encoders.fit_and_preprocess_audio(
            dataset_audio, device
        )
        torch.save(two_encoders, model_path)

    dataset = torch.utils.data.TensorDataset(dataset_settings, dataset_reduced_audio)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    with torch.inference_mode():
        test_settings = torch.randn(
            test_dataset_size, settings_encoder_input_size, device=device
        )
        test_audio = synth_parallel(
            test_settings, torch.linspace(0, num_seconds, num_samples, device=device)
        )
        test_audio = two_encoders.preprocess_audio(test_audio)

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
            for settings, audio in test_dataloader:
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

    for epoch in range(num_epochs):
        two_encoders.train()
        losses = []

        for settings, audio in dataloader:
            optimizer.zero_grad()

            loss = two_encoders.loss(audio.to(device), settings.to(device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                two_encoders.parameters(),
                max_norm=gradient_clip_value,
                error_if_nonfinite=True,
                foreach=foreach,
            )
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch: {epoch}, average loss: {np.mean(losses)}")

        if (epoch + 1) % evaluate_every_n_epochs == 0:
            two_encoders.eval()

            with torch.inference_mode():
                eval_loss, eval_audio_accuracy, eval_settings_accuracy = evaluate()
                print(
                    f"Epoch: {epoch}, evaluation loss: {eval_loss}, audio accuracy: {eval_audio_accuracy}, settings accuracy: {eval_settings_accuracy}"
                )

                print(f"log_t: {two_encoders.log_t.item()}, b: {two_encoders.b.item()}")

        if (epoch + 1) % save_every_n_epochs == 0:
            print("Saving model")
            two_encoders.eval()

            torch.save(two_encoders, model_path)
            torch.save(optimizer, optimizer_path)
            print("Done saving model and optimizer state")

        if (epoch + 1) % regenerate_dataset_every_n_epochs == 0:
            print("Regenerating dataset")
            dataset = torch.utils.data.TensorDataset(
                *generate_dataset(dataset_size, torch.device("cpu"), device)
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
        
        if (epoch + 1) % generate_sound_every_n_epochs == 0:
            print("Generating sound")
            time = torch.linspace(0, num_seconds, num_samples, device=device).unsqueeze(0)
            target_audio = torch.sin(2 * torch.pi * time * 440)
            settings = two_encoders.reconstruct_settings(target_audio)
            with torch.inference_mode():
                eval_time = torch.linspace(0, num_seconds, num_seconds * 48000, device=device)
                eval_audio = synth_parallel(settings, eval_time)
                # save eval_audio to wav file
                sf.write("eval_audio.wav", eval_audio.cpu().numpy()[0], 48000)


    two_encoders.eval()

    torch.save(two_encoders, model_path)


if __name__ == "__main__":
    train()
