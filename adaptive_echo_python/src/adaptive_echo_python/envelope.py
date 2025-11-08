import torch
import torch.nn as nn

from adaptive_echo_python.interpolation import exp_interp, linear_interp


# envolope generator
def env(
    time: torch.Tensor,
    length: torch.Tensor,
    attack: torch.Tensor,
    decay: torch.Tensor,
    sustain: torch.Tensor,
    release: torch.Tensor,
) -> torch.Tensor:
    value = torch.where(
        time < attack,
        time / attack,
        torch.where(
            time < attack + decay,
            1.0 - (1.0 - sustain) * (time - attack) / decay,
            torch.where(
                time < length - release, sustain, sustain * (length - time) / release
            ),
        ),
    )

    value = torch.clamp(value, min=0.0, max=1.0)
    return value


# use envelope generator with inputs from 0 to 1
def env_uniform(
    time: torch.Tensor,
    length: torch.Tensor,
    attack: torch.Tensor,
    decay: torch.Tensor,
    sustain: torch.Tensor,
    release: torch.Tensor,
) -> torch.Tensor:
    min_length = torch.tensor(0.1, dtype=time.dtype, device=time.device)
    max_length = torch.tensor(1.0, dtype=time.dtype, device=time.device)
    length = exp_interp(min_length, max_length, length)

    min_attack = torch.tensor(0.05, dtype=time.dtype, device=time.device)
    max_attack = torch.tensor(0.5, dtype=time.dtype, device=time.device)
    attack = exp_interp(min_attack, max_attack, attack)

    min_decay = torch.tensor(0.05, dtype=time.dtype, device=time.device)
    max_decay = torch.tensor(0.5, dtype=time.dtype, device=time.device)
    decay = exp_interp(min_decay, max_decay, decay)

    min_sustain = torch.tensor(0.1, dtype=time.dtype, device=time.device)
    max_sustain = torch.tensor(1.0, dtype=time.dtype, device=time.device)
    sustain = linear_interp(min_sustain, max_sustain, sustain)

    min_release = torch.tensor(0.05, dtype=time.dtype, device=time.device)
    max_release = torch.tensor(0.5, dtype=time.dtype, device=time.device)
    release = exp_interp(min_release, max_release, release)

    return env(time, length, attack, decay, sustain, release)


class EnvelopeUniform(nn.Module):
    def __init__(self):
        super(EnvelopeUniform, self).__init__()
        self.length = nn.Parameter(torch.tensor(0.0))
        self.attack = nn.Parameter(torch.tensor(0.0))
        self.decay = nn.Parameter(torch.tensor(0.0))
        self.sustain = nn.Parameter(torch.tensor(0.0))
        self.release = nn.Parameter(torch.tensor(0.0))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return env_uniform(
            time, self.length, self.attack, self.decay, self.sustain, self.release
        )
