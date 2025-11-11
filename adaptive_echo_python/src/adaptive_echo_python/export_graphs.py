from pathlib import Path

import torch
import torch.nn as nn

from adaptive_echo_python.synth import Synth
from adaptive_echo_python.two_encoders import TwoEncoders


def export_torchscript(path: Path, model: nn.Module):
    scripted_model = torch.jit.script(model)
    scripted_model.save(str(path))


def export_graphs():
    synth = Synth()
    settings_input = synth.encode_settings()

    export_torchscript(
        Path("../plugin/synth.pt"),
        synth,
    )

    audio_encoder_input_size = 48000 * 5  # 5 seconds
    settings_encoder_input_size = settings_input.shape[0]
    encoder_embedding_size = 256
    encoder_hidden_size = 256
    encoder_num_layers = 6

    two_encoders = TwoEncoders(
        audio_encoder_input_size,
        settings_encoder_input_size,
        encoder_embedding_size,
        encoder_hidden_size,
        encoder_num_layers,
    )

    # Export as TorchScript format (.pt) for C++ compatibility
    export_torchscript(
        Path("../plugin/two_encoders.pt"),
        two_encoders,
    )


if __name__ == "__main__":
    graphs_path = Path("graphs")
    if not graphs_path.exists():
        graphs_path.mkdir(parents=True)
    export_graphs()
