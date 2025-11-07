from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.export import ExportedProgram, export, save

from adaptive_echo_python.encoder import Encoder


def export_graph(path: Path, target: Callable | nn.Module, args: tuple):
    program: ExportedProgram = export(target, args)
    save(program, path)


def export_graphs():
    audio_encoder_input_size = 48000 * 5  # 5 seconds
    encoder_embedding_size = 256
    encoder_hidden_size = 256
    encoder_num_layers = 6
    export_graph(
        Path("graphs/audio_encoder.pt2"),
        Encoder(
            audio_encoder_input_size,
            encoder_embedding_size,
            encoder_hidden_size,
            encoder_num_layers,
        ),
        (torch.randn(1, audio_encoder_input_size),),
    )


if __name__ == "__main__":
    graphs_path = Path("graphs")
    if not graphs_path.exists():
        graphs_path.mkdir(parents=True)
    export_graphs()
