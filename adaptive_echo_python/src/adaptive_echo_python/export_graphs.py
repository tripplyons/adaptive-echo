from pathlib import Path

import torch
import torch.nn as nn

from adaptive_echo_python.encoder import Encoder


def export_torchscript(path: Path, model: nn.Module, example_input: torch.Tensor):
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(str(path))

    with torch.inference_mode():
        output = traced_model(example_input)
        print("output.shape:", output.shape)


def export_graphs():
    audio_encoder_input_size = 48000 * 5  # 5 seconds
    encoder_embedding_size = 256
    encoder_hidden_size = 256
    encoder_num_layers = 6

    model = Encoder(
        audio_encoder_input_size,
        encoder_embedding_size,
        encoder_hidden_size,
        encoder_num_layers,
    )

    example_input_shape = (1, audio_encoder_input_size)
    example_input = torch.randn(example_input_shape)

    # Export as TorchScript format (.pt) for C++ compatibility
    export_torchscript(
        Path("graphs/audio_encoder.pt"),
        model,
        example_input,
    )


if __name__ == "__main__":
    graphs_path = Path("graphs")
    if not graphs_path.exists():
        graphs_path.mkdir(parents=True)
    export_graphs()
