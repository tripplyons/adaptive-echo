import wave

import jax.numpy as jnp
import numpy as np


def save_wav(filename, data: jnp.ndarray, sample_rate):
    assert data.ndim == 1
    assert jnp.abs(data).max() <= 1
    assert data.dtype == jnp.float32
    data *= 2**15
    data = np.array(data.astype(jnp.int16), dtype=np.int16)
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.setnframes(data.shape[0])
        wav_file.writeframes(data.tobytes())


def load_wav(filename) -> (jnp.ndarray, int):
    with wave.open(filename, "rb") as wav_file:
        data = wav_file.readframes(wav_file.getnframes())
        data = np.frombuffer(data, dtype=np.int16)
        data = data.astype(jnp.float32) / 2**15
        sample_rate = wav_file.getframerate()
        return jnp.array(data), sample_rate


def normalize_wav(data: jnp.ndarray) -> jnp.ndarray:
    assert data.dtype == jnp.float32
    data = data - data.mean(axis=-1, keepdims=True)
    max_abs = jnp.max(jnp.abs(data), axis=-1, keepdims=True)
    headroom = 0.95
    data = data / max_abs * headroom

    return data
