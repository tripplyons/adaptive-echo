import math

import torch
import torch.nn as nn

from adaptive_echo_python.interpolation import exp_interp, linear_interp


# waveform + noise generator
def osc(
    # time
    time: torch.Tensor,
    # frequency
    freq: torch.Tensor,
    # phase shift
    phase_shift: torch.Tensor,
    # amount of even harmonics (one axis of the wavetable)
    warmth: torch.Tensor,
    # amount of higher harmonics (another axis of the wavetable)
    harshness: torch.Tensor,
    # overall volume
    amplitude: torch.Tensor,
    # amount of noise
    noise_level: torch.Tensor,
    # signal for frequency modulation
    modulation: torch.Tensor | None = None,
    # amount of frequency modulation
    fm_amount: torch.Tensor | None = None,
):
    noise = torch.randn(time.shape, dtype=time.dtype, device=time.device) * 0.2

    phase = time * freq + phase_shift
    if modulation is not None and fm_amount is not None:
        phase += modulation * fm_amount
    phase %= 1

    phase = 0.5 * (phase**warmth - (1 - phase) ** warmth + 1)

    phase *= 2 * torch.pi

    sin = torch.sin(phase)

    wave = torch.sign(sin) * torch.abs(sin) ** harshness * amplitude

    noise_interp = 0.2 * noise_level

    return linear_interp(wave, noise, noise_interp)


def osc_uniform(
    time: torch.Tensor,
    freq: torch.Tensor,
    phase_shift: torch.Tensor,
    warmth: torch.Tensor,
    harshness: torch.Tensor,
    amplitude: torch.Tensor,
    noise_level: torch.Tensor,
    modulation: torch.Tensor | None = None,
    fm_amount: torch.Tensor | None = None,
):
    min_freq = math.log2(20.0) * 12
    max_freq = math.log2(5000.0) * 12
    semitones = linear_interp(min_freq, max_freq, freq)
    freq = 2 ** (semitones / 12)

    min_phase_shift = 0
    max_phase_shift = 1
    phase_shift = linear_interp(min_phase_shift, max_phase_shift, phase_shift)

    min_warmth = 1 / 5
    max_warmth = 5
    warmth = exp_interp(min_warmth, max_warmth, warmth)

    min_harshness = 1 / 5
    max_harshness = 5
    harshness = exp_interp(min_harshness, max_harshness, harshness)

    min_amplitude = 0.1
    max_amplitude = 1
    amplitude = linear_interp(min_amplitude, max_amplitude, amplitude)

    return osc(
        time=time,
        freq=freq,
        phase_shift=phase_shift,
        warmth=warmth,
        harshness=harshness,
        amplitude=amplitude,
        noise_level=noise_level,
        modulation=modulation,
        fm_amount=fm_amount,
    )


class OscillatorModulatedUniform(nn.Module):
    def __init__(self):
        super(OscillatorModulatedUniform, self).__init__()
        self.low_freq = nn.Parameter(torch.tensor(0.0))
        self.high_freq = nn.Parameter(torch.tensor(0.0))
        self.low_phase_shift = nn.Parameter(torch.tensor(0.0))
        self.high_phase_shift = nn.Parameter(torch.tensor(0.0))
        self.low_warmth = nn.Parameter(torch.tensor(0.0))
        self.high_warmth = nn.Parameter(torch.tensor(0.0))
        self.low_harshness = nn.Parameter(torch.tensor(0.0))
        self.high_harshness = nn.Parameter(torch.tensor(0.0))
        self.low_amplitude = nn.Parameter(torch.tensor(0.0))
        self.high_amplitude = nn.Parameter(torch.tensor(0.0))
        self.low_noise_level = nn.Parameter(torch.tensor(0.0))
        self.high_noise_level = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        time: torch.Tensor,
        low_high_interp: torch.Tensor,
        modulation=None,
        fm_amount=None,
    ) -> torch.Tensor:
        return osc_uniform(
            time,
            linear_interp(self.low_freq, self.high_freq, low_high_interp),
            linear_interp(self.low_phase_shift, self.high_phase_shift, low_high_interp),
            linear_interp(self.low_warmth, self.high_warmth, low_high_interp),
            linear_interp(self.low_harshness, self.high_harshness, low_high_interp),
            linear_interp(self.low_amplitude, self.high_amplitude, low_high_interp),
            linear_interp(self.low_noise_level, self.high_noise_level, low_high_interp),
            modulation=modulation,
            fm_amount=fm_amount,
        )

    def encode_settings(self) -> torch.Tensor:
        return torch.cat(
            [
                self.low_freq.view((1,)),
                self.high_freq.view((1,)),
                self.low_phase_shift.view((1,)),
                self.high_phase_shift.view((1,)),
                self.low_warmth.view((1,)),
                self.high_warmth.view((1,)),
                self.low_harshness.view((1,)),
                self.high_harshness.view((1,)),
                self.low_amplitude.view((1,)),
                self.high_amplitude.view((1,)),
                self.low_noise_level.view((1,)),
                self.high_noise_level.view((1,)),
            ]
        )
