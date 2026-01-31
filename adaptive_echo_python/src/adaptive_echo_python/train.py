from pathlib import Path

import numpy as np
import torch
try:
    from schedulefree import RAdamScheduleFree
except ImportError:
    RAdamScheduleFree = None

from adaptive_echo_python.synth import Synth, synth_parallel
from adaptive_echo_python.two_encoders import TwoEncoders

num_seconds = 2
training_sample_rate = 16384
num_samples = training_sample_rate * num_seconds
settings_encoder_input_size = Synth().encode_settings().shape[0]
reduced_audio_size = 8192
encoder_embedding_size = 2048
encoder_hidden_size = 2048
encoder_num_layers = 2
learning_rate = 1e-3
gradient_clip_value = 10.0
batch_size = 1024
steps_per_epoch = 10
evaluation_batch_size = 1000
test_steps = 5
num_epochs = 1000000  # basically forever
evaluate_every_n_epochs = 10
save_every_n_epochs = 10
settings_scale = 1.5
model_path = Path("./two_encoders.pt")
optimizer_path = Path("./two_encoders_optimizer.pt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_batch(device, synth_device, preprocess_fn):
    """Generate a single batch of training data on the fly."""
    with torch.inference_mode():
        new_settings = settings_scale * torch.randn(
            batch_size, settings_encoder_input_size, device=synth_device
        )
        new_audio = synth_parallel(
            torch.sigmoid(new_settings),
            torch.linspace(0, num_seconds, num_samples, device=synth_device),
        )
        new_audio = preprocess_fn(new_audio)

    return new_settings.to(device, non_blocking=True), new_audio.to(
        device, non_blocking=True
    )


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

    optimizer = RAdamScheduleFree(
        two_encoders.parameters(),
        lr=learning_rate,
    )

    two_encoders.train()
    optimizer.train()

    # Generate initial data to fit the audio reduction encoder
    def generate_preprocessed_batch(synth_device):
        """Generate a batch and preprocess it for fitting the audio reduction encoder."""
        with torch.inference_mode():
            new_settings = settings_scale * torch.randn(
                batch_size, settings_encoder_input_size, device=synth_device
            )
            new_audio = synth_parallel(
                torch.sigmoid(new_settings),
                torch.linspace(0, num_seconds, num_samples, device=synth_device),
            )
        return new_audio

    # Generate initial data for fitting (non-preprocessed)
    initial_audio_batch = generate_preprocessed_batch(device)

    two_encoders = torch.compile(two_encoders)

    if existing_model:
        with torch.inference_mode():
            _ = two_encoders.preprocess_audio(initial_audio_batch.to(device))
    else:
        two_encoders.train()
        optimizer.train()
        # Generate multiple batches for fitting the audio reduction encoder
        fit_audio_batches = []
        for _ in range(10):
            fit_audio_batches.append(generate_preprocessed_batch(device).cpu())
        fit_audio = torch.cat(fit_audio_batches, dim=0)
        _ = two_encoders.fit_and_preprocess_audio(fit_audio, device)
        torch.save(two_encoders, model_path)

    # Generate test data once (kept in CPU memory)
    with torch.inference_mode():
        test_settings_list = []
        test_audio_list = []
        for _ in range(test_steps):
            test_settings = settings_scale * torch.randn(
                evaluation_batch_size, settings_encoder_input_size, device=device
            )
            test_audio = synth_parallel(
                torch.sigmoid(test_settings),
                torch.linspace(0, num_seconds, num_samples, device=device),
            )
            test_audio = two_encoders.preprocess_audio(test_audio)
            test_settings_list.append(test_settings.cpu())
            test_audio_list.append(test_audio.cpu())

        test_settings = torch.cat(test_settings_list, dim=0)
        test_audio = torch.cat(test_audio_list, dim=0)

    def evaluate():
        two_encoders.eval()
        optimizer.eval()
        with torch.inference_mode():
            losses = []
            settings_accuracies = []
            audio_accuracies = []
            # Evaluate in batches to avoid memory issues
            for i in range(0, len(test_audio), evaluation_batch_size):
                settings_batch = test_settings[i : i + evaluation_batch_size].to(device)
                audio_batch = test_audio[i : i + evaluation_batch_size].to(device)
                loss = two_encoders.loss(audio_batch, settings_batch)
                losses.append(loss.item())
                audio_accuracy, settings_accuracy = two_encoders.accuracies(
                    audio_batch, settings_batch
                )
                audio_accuracies.append(audio_accuracy.item())
                settings_accuracies.append(settings_accuracy.item())

            return (
                np.mean(losses),
                np.mean(audio_accuracies),
                np.mean(settings_accuracies),
            )

    print("Starting training with on-the-fly data generation")

    eval_loss, eval_audio_accuracy, eval_settings_accuracy = evaluate()
    print(
        f"Initial evaluation loss: {eval_loss}, audio accuracy: {eval_audio_accuracy}, settings accuracy: {eval_settings_accuracy}"
    )

    for epoch in range(num_epochs):
        two_encoders.train()
        optimizer.train()
        losses = []

        # Generate and train on batches on the fly
        for _ in range(steps_per_epoch):
            settings_batch, audio_batch = generate_batch(
                device, device, two_encoders.preprocess_audio
            )

            optimizer.zero_grad()

            loss = two_encoders.loss(audio_batch, settings_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                two_encoders.parameters(),
                max_norm=gradient_clip_value,
                error_if_nonfinite=True,
            )
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch: {epoch}, average loss: {np.mean(losses)}")

        if (epoch + 1) % evaluate_every_n_epochs == 0:
            two_encoders.eval()
            optimizer.eval()

            with torch.inference_mode():
                eval_loss, eval_audio_accuracy, eval_settings_accuracy = evaluate()
                print(
                    f"Epoch: {epoch}, evaluation loss: {eval_loss}, audio accuracy: {eval_audio_accuracy}, settings accuracy: {eval_settings_accuracy}"
                )

                print(f"log_t: {two_encoders.log_t.item()}, b: {two_encoders.b.item()}")

        if (epoch + 1) % save_every_n_epochs == 0:
            print("Saving model")
            two_encoders.eval()
            optimizer.eval()

            torch.save(two_encoders, model_path)
            torch.save(optimizer, optimizer_path)
            print("Done saving model and optimizer state")

    two_encoders.eval()
    optimizer.eval()

    torch.save(two_encoders, model_path)
    torch.save(optimizer, optimizer_path)


if __name__ == "__main__":
    train()
