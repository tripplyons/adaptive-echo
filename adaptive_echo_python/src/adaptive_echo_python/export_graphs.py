from pathlib import Path

import torch
import torch.nn as nn

from adaptive_echo_python.synth import Synth


def export_torchscript(path: Path, model: nn.Module):
    scripted_model = torch.jit.script(model)
    scripted_model.save(str(path))


def export_graphs():
    synth = Synth()

    export_torchscript(
        Path("../plugin/synth.pt"),
        synth,
    )


if __name__ == "__main__":
    graphs_path = Path("graphs")
    if not graphs_path.exists():
        graphs_path.mkdir(parents=True)
    export_graphs()
