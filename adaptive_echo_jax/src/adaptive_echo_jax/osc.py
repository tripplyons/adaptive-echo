"""
Oscillator functions for jax.
"""

from typing import Optional

import jax
import jax.numpy as np

from adaptive_echo_jax.interpolation import exp_interp, linear_interp


def osc(
    time: np.ndarray,
    freq: np.ndarray,
    phase_shift: np.ndarray,
    warmth: np.ndarray,
    harshness: np.ndarray,
    amplitude: np.ndarray,
    noise_level: np.ndarray,
    modulation: Optional[np.ndarray] = None,
    fm_amount: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate oscillator waveform with optional frequency modulation."""
    EPSILON = 1e-6

    # JAX requires explicit PRNG key for random operations
    # Use a fixed key for deterministic noise (noise level is controlled by noise_level parameter)
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, time.shape, dtype=time.dtype) * 0.5

    phase = time * freq + phase_shift
    if modulation is not None and fm_amount is not None:
        phase += modulation * fm_amount
    phase = phase % 1

    phase = np.clip(phase, EPSILON, 1.0 - EPSILON)

    phase_pow = np.power(phase, warmth)
    one_minus_phase_pow = np.power(1.0 - phase, warmth)

    phase_pow = np.where(np.isfinite(phase_pow), phase_pow, np.zeros_like(phase_pow))
    one_minus_phase_pow = np.where(
        np.isfinite(one_minus_phase_pow),
        one_minus_phase_pow,
        np.zeros_like(one_minus_phase_pow),
    )
    phase = 0.5 * (phase_pow - one_minus_phase_pow + 1)

    phase *= 2 * np.pi

    sin = np.sin(phase)

    abs_sin = np.abs(sin)
    abs_sin = np.clip(abs_sin, EPSILON, 1.0)
    sin_pow = np.power(abs_sin, harshness)

    sin_pow = np.where(np.isfinite(sin_pow), sin_pow, np.zeros_like(sin_pow))

    wave = np.sign(sin) * sin_pow * amplitude

    noise_interp = 0.1 * noise_level

    return linear_interp(wave, noise, noise_interp)


def osc_uniform(
    time: np.ndarray,
    freq: np.ndarray,
    phase_shift: np.ndarray,
    warmth: np.ndarray,
    harshness: np.ndarray,
    amplitude: np.ndarray,
    noise_level: np.ndarray,
    modulation: Optional[np.ndarray] = None,
    fm_amount: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate oscillator waveform with inputs normalized to [0, 1]."""
    min_freq = np.log2(np.array(50.0, dtype=time.dtype)) * 12
    max_freq = np.log2(np.array(2000.0, dtype=time.dtype)) * 12
    semitones = linear_interp(min_freq, max_freq, freq)
    freq = np.power(2.0, semitones / 12.0)

    min_phase_shift = np.array(0, dtype=time.dtype)
    max_phase_shift = np.array(1.0, dtype=time.dtype)
    phase_shift = linear_interp(min_phase_shift, max_phase_shift, phase_shift)

    min_warmth = np.array(1.0 / 5.0, dtype=time.dtype)
    max_warmth = np.array(5.0, dtype=time.dtype)
    warmth = exp_interp(min_warmth, max_warmth, warmth)

    min_harshness = np.array(1.0 / 5.0, dtype=time.dtype)
    max_harshness = np.array(5.0, dtype=time.dtype)
    harshness = exp_interp(min_harshness, max_harshness, harshness)

    min_amplitude = np.array(0.1, dtype=time.dtype)
    max_amplitude = np.array(1.0, dtype=time.dtype)
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
