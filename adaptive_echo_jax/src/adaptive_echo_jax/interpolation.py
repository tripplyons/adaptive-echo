"""
Interpolation functions for jax.
"""

import jax.numpy as np


def linear_interp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation between a and b using t as the interpolation factor."""
    return a + (b - a) * t


def exp_interp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Exponential interpolation between a and b using t as the interpolation factor."""
    EPSILON = 1e-6

    a_clamped = np.clip(np.abs(a), EPSILON, None) * np.sign(a)
    b_clamped = np.clip(np.abs(b), EPSILON, None) * np.sign(b)

    ratio = b_clamped / a_clamped
    ratio = np.clip(ratio, EPSILON, 1.0 / EPSILON)

    return a_clamped * np.power(ratio, t)
