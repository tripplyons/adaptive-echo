import os

import jax.numpy as jnp

from adaptive_echo.audio import load_wav, save_wav


def test_save_wav():
    data = jnp.zeros(1000, dtype=jnp.float32)
    save_wav("test_save_wav.wav", data, 44100)
    os.remove("test_save_wav.wav")


def test_load_wav():
    save_wav("test_load_wav.wav", jnp.zeros(1000, dtype=jnp.float32), 44100)
    data, sample_rate = load_wav("test_load_wav.wav")
    assert sample_rate == 44100
    assert data.shape == (1000,)
    assert data.mean() == 0
    assert data.std() == 0
    os.remove("test_load_wav.wav")
