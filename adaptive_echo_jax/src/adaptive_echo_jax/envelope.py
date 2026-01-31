"""
Envelope generator functions for jax.
"""

import jax.numpy as np

from adaptive_echo_jax.interpolation import exp_interp, linear_interp


def env(
    time: np.ndarray,
    length: np.ndarray,
    attack: np.ndarray,
    decay: np.ndarray,
    sustain: np.ndarray,
    release: np.ndarray,
) -> np.ndarray:
    """Generate an ADSR envelope."""
    value = np.where(
        time < attack,
        time / attack,
        np.where(
            time < attack + decay,
            1.0 - (1.0 - sustain) * (time - attack) / decay,
            np.where(
                time < length - release,
                sustain,
                np.where(time < length, sustain * (length - time) / release, 0),
            ),
        ),
    )

    return value


def env_uniform(
    time: np.ndarray,
    length: np.ndarray,
    attack: np.ndarray,
    decay: np.ndarray,
    sustain: np.ndarray,
    release: np.ndarray,
) -> np.ndarray:
    """Generate an ADSR envelope with inputs normalized to [0, 1]."""
    min_length = np.array(0.2, dtype=time.dtype)
    max_length = np.array(2.0, dtype=time.dtype)
    length = exp_interp(min_length, max_length, length)

    min_attack = np.array(0.05, dtype=time.dtype)
    max_attack = np.array(0.5, dtype=time.dtype)
    attack = exp_interp(min_attack, max_attack, attack)

    min_decay = np.array(0.05, dtype=time.dtype)
    max_decay = np.array(0.5, dtype=time.dtype)
    decay = exp_interp(min_decay, max_decay, decay)

    min_sustain = np.array(0.1, dtype=time.dtype)
    max_sustain = np.array(1.0, dtype=time.dtype)
    sustain = linear_interp(min_sustain, max_sustain, sustain)

    min_release = np.array(0.05, dtype=time.dtype)
    max_release = np.array(0.5, dtype=time.dtype)
    release = exp_interp(min_release, max_release, release)

    return env(time, length, attack, decay, sustain, release)
