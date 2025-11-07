import os

import jax.numpy as jnp

from adaptive_echo.audio import load_wav, save_wav


# make sure the file is saved
def test_save_wav():
    data = jnp.zeros(1000, dtype=jnp.float32)
    save_wav("test_save_wav.wav", data, 44100)

    assert os.path.exists("test_save_wav.wav")

    # cleanup
    os.remove("test_save_wav.wav")


# make sure the file can be loaded
def test_load_wav():
    save_wav("test_load_wav.wav", jnp.zeros(1000, dtype=jnp.float32), 44100)
    data, sample_rate = load_wav("test_load_wav.wav")

    # make sure the data is the same
    assert sample_rate == 44100
    assert data.shape == (1000,)
    assert data.mean() == 0
    assert data.std() == 0

    # cleanup
    os.remove("test_load_wav.wav")
