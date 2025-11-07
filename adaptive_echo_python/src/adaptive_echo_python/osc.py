import torch

from adaptive_echo_python.interpolation import exp_interp, linear_interp


# waveform + noise generator
def osc(
    # random number generator (for noise)
    rng,
    # time
    time: torch.Tensor[float],
    # frequency
    freq: torch.Tensor[float],
    # phase shift
    phase_shift: torch.Tensor[float],
    # amount of even harmonics (one axis of the wavetable)
    warmth,
    # amount of higher harmonics (another axis of the wavetable)
    harshness,
    # overall volume
    amplitude,
    # amount of noise
    noise_level,
    # signal for frequency modulation
    modulation=None,
    # amount of frequency modulation
    fm_amount=0,
):
    noise = torch.rand(time.shape)

    phase = time * freq + phase_shift
    if modulation is not None:
        phase += modulation * fm_amount
    phase %= 1

    phase = 0.5 * (phase**warth - (1 - phase) ** warmth + 1)

    phase *= 2 * torch.pi

    sin = torch.sin(phase)

    wave = torch.sign(sin) * torch.abs(sin) ** harshness * amplitude

    noise_interp = 0.2 * noise_level

    return linear_interp(wave, noice, noise_interp)


def osc_uniform(
    rng,
    time,
    freq,
    phase_shift,
    warmth,
    harshness,
    amplitude,
    noise_level,
    modulation=None,
    fm_amount=0,
):
    min_freq = torch.log2(10) * 12
    max_freq = torch.log2(20000) * 12
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
        rng,
        time,
        freq,
        phase_shift,
        warmth,
        harshness,
        amplitude,
        noise_level,
        modulation,
        fm_amount,
    )
