"""
Synthesizer functions for jax.
"""

import jax
import jax.numpy as np

from adaptive_echo_jax.envelope import env_uniform
from adaptive_echo_jax.interpolation import linear_interp
from adaptive_echo_jax.osc import osc_uniform


@jax.jit
def synth(settings: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Generate audio from synthesizer settings.
    JIT-compiled for optimal performance.
    """
    env_vol_a = env_uniform(
        times, settings[0], settings[1], settings[2], settings[3], settings[4]
    )
    env_vol_b = env_uniform(
        times, settings[5], settings[6], settings[7], settings[8], settings[9]
    )
    env_mod = env_uniform(
        times, settings[10], settings[11], settings[12], settings[13], settings[14]
    )
    env_fm = env_uniform(
        times, settings[39], settings[40], settings[41], settings[42], settings[43]
    )

    fm_range_low = settings[44]
    fm_range_high = settings[45]
    fm_amount = linear_interp(fm_range_low, fm_range_high, env_fm[None])[0]

    # Optimized: extract settings once
    osc_b_low_settings = settings[27:39:2]
    osc_b_high_settings = settings[28:39:2]
    osc_b_settings = linear_interp(
        osc_b_low_settings[..., None], osc_b_high_settings[..., None], env_mod
    )[..., 0]
    osc_b = osc_uniform(
        times,
        osc_b_settings[0],
        osc_b_settings[1],
        osc_b_settings[2],
        osc_b_settings[3],
        osc_b_settings[4],
        osc_b_settings[5],
    )

    osc_a_low_settings = settings[15:27:2]
    osc_a_high_settings = settings[16:27:2]
    osc_a_settings = linear_interp(
        osc_a_low_settings[..., None], osc_a_high_settings[..., None], env_mod
    )[..., 0]
    osc_a = osc_uniform(
        times,
        osc_a_settings[0],
        osc_a_settings[1],
        osc_a_settings[2],
        osc_a_settings[3],
        osc_a_settings[4],
        osc_a_settings[5],
        modulation=osc_b,
        fm_amount=fm_amount,
    )

    # Optimized: single multiply-add operation
    return osc_a * env_vol_a + osc_b * env_vol_b


@jax.jit
def synth_parallel(settings: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Generate audio for multiple settings in parallel using vmap."""
    return jax.vmap(synth, in_axes=(0, None))(settings, times)
