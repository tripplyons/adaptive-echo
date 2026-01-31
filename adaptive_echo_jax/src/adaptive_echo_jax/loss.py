"""
STFT-based audio loss functions for JAX.
Fast multi-resolution spectral similarity measurement.
"""

from typing import Tuple
import functools

import jax
import jax.numpy as np
from jax.scipy import signal
import numpy as numpy_lib


def _stft_single(x: np.ndarray, win_length: int, hop_length: int, n_fft: int, window: np.ndarray) -> np.ndarray:
    """Single STFT computation."""
    _, _, stft_result = signal.stft(
        x,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        window=window,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    return stft_result


# Vectorized STFT using vmap (for single device)
_stft_batch = jax.vmap(_stft_single, in_axes=(0, None, None, None, None))


@jax.jit(static_argnames=['fft_sizes', 'hop_sizes', 'sample_rate'])
def fast_audio_loss_precomputed(
    generated: np.ndarray,
    target_stfts: Tuple[np.ndarray, ...],
    target_zcr: np.ndarray,
    fft_sizes: Tuple[int, ...] = (1024, 512, 256),
    hop_sizes: Tuple[int, ...] = (512, 256, 128),
    sample_rate: float = 16384.0,
) -> np.ndarray:
    """
    Fast audio similarity loss with precomputed target features.

    Uses multi-resolution STFT with:
    - Spectral convergence loss (70% weight)
    - Log-magnitude loss (30% weight)
    - Zero-crossing rate loss (5% weight)

    This matches the PyTorch fast_audio_loss implementation.

    Args:
        generated: Generated audio [batch_size, num_samples]
        target_stfts: Precomputed target STFT magnitudes (one per scale)
        target_zcr: Precomputed target zero-crossing rate
        fft_sizes: FFT window sizes (default: [1024, 512, 256])
        hop_sizes: Hop lengths (default: [512, 256, 128])
        sample_rate: Sample rate (default: 16384.0)

    Returns:
        Loss array [batch_size]
    """
    # Normalize generated audio
    gen_mean = np.mean(generated, axis=-1, keepdims=True)
    gen_norm = generated - gen_mean
    gen_std = np.std(gen_norm, axis=-1, keepdims=True)
    gen_norm = gen_norm / (gen_std + 1e-8)

    batch_size = generated.shape[0]
    total_loss = np.zeros(batch_size)

    # Compute STFT losses with different FFT sizes
    for i, (n_fft, hop_length) in enumerate(zip(fft_sizes, hop_sizes)):
        win_length = n_fft
        window = np.array(numpy_lib.hanning(win_length).astype(numpy_lib.float32))

        # STFT for generated
        x_stft = _stft_batch(gen_norm, win_length, hop_length, n_fft, window)
        x_mag = np.abs(x_stft)

        # Get precomputed target magnitude
        y_mag = target_stfts[i]
        if y_mag.ndim == 2:
            y_mag = y_mag[np.newaxis, :, :]

        # Fast spectral convergence using mean instead of Frobenius norm
        y_mag_mean = np.mean(y_mag, axis=(1, 2), keepdims=True)
        sc_loss = np.mean(
            np.abs(y_mag - x_mag) / (y_mag_mean + 1e-8),
            axis=(1, 2),
        )

        # Log-magnitude loss
        log_y = np.log(y_mag + 1e-8)
        log_x = np.log(x_mag + 1e-8)
        mag_loss = np.mean(np.abs(log_y - log_x), axis=(1, 2))

        # Weight the losses (spectral convergence 70%, log-magnitude 30%)
        total_loss += 0.7 * sc_loss + 0.3 * mag_loss

    # Average over scales
    total_loss /= len(fft_sizes)

    # Add zero-crossing rate loss
    gen_zcr = ((gen_norm[:, 1:] * gen_norm[:, :-1]) < 0).astype(np.float32).mean(axis=-1)
    zcr_loss = np.abs(gen_zcr - target_zcr)

    # Combine with small weight (95% STFT, 5% ZCR)
    total_loss = 0.95 * total_loss + 0.05 * zcr_loss

    return total_loss


def precompute_target_stft(
    target: np.ndarray,
    fft_sizes: Tuple[int, ...] = (1024, 512, 256),
    hop_sizes: Tuple[int, ...] = (512, 256, 128),
    sample_rate: float = 16384.0,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Precompute target STFT magnitudes and ZCR for fast_audio_loss.

    Args:
        target: Target audio [num_samples]
        fft_sizes: FFT window sizes (default: [1024, 512, 256])
        hop_sizes: Hop lengths (default: [512, 256, 128])
        sample_rate: Sample rate (default: 16384.0)

    Returns:
        Tuple of:
        - target_stfts: Tuple of STFT magnitudes (one per scale)
        - target_zcr: Zero-crossing rate
        - normalized_target: Normalized target audio
    """
    # Normalize target
    tgt_mean = np.mean(target)
    tgt_norm = target - tgt_mean
    tgt_std = np.std(tgt_norm)
    normalized_target = tgt_norm / (tgt_std + 1e-8)

    # Compute STFT magnitudes for each scale
    target_stfts = []
    for n_fft, hop_length in zip(fft_sizes, hop_sizes):
        win_length = n_fft
        window = np.array(numpy_lib.hanning(win_length).astype(numpy_lib.float32))
        y_stft = _stft_single(normalized_target, win_length, hop_length, n_fft, window)
        y_mag = np.abs(y_stft)
        target_stfts.append(y_mag)

    # Compute zero-crossing rate
    target_zcr = ((normalized_target[1:] * normalized_target[:-1]) < 0).astype(np.float32).mean()

    return tuple(target_stfts), target_zcr, normalized_target


def fast_audio_loss(
    target: np.ndarray,
    fft_sizes: Tuple[int, ...] = (1024, 512, 256),
    hop_sizes: Tuple[int, ...] = (512, 256, 128),
    sample_rate: float = 16384.0,
):
    """
    Create a JIT-compiled fast audio loss function with precomputed target features.

    This matches the PyTorch fast_audio_loss implementation and uses:
    - Multi-resolution STFT (3 scales by default)
    - Spectral convergence loss (70%)
    - Log-magnitude loss (30%)
    - Zero-crossing rate loss (5%)

    Usage:
        loss_fn = fast_audio_loss(target_audio)
        losses = loss_fn(generated_audio)

    Args:
        target: Target audio [num_samples]
        fft_sizes: FFT window sizes (default: [1024, 512, 256])
        hop_sizes: Hop lengths (default: [512, 256, 128])
        sample_rate: Sample rate (default: 16384.0)

    Returns:
        A JIT-compiled function: fn(generated: np.ndarray) -> np.ndarray
    """
    # Precompute target features
    target_stfts, target_zcr, _ = precompute_target_stft(
        target, fft_sizes, hop_sizes, sample_rate
    )

    return functools.partial(
        fast_audio_loss_precomputed,
        target_stfts=target_stfts,
        target_zcr=target_zcr,
        fft_sizes=fft_sizes,
        hop_sizes=hop_sizes,
        sample_rate=sample_rate,
    )


def combined_loss(
    target: np.ndarray,
    stft_weight: float = 1.0,
    stft_fft_sizes: Tuple[int, ...] = (1024, 512, 256),
    stft_hop_sizes: Tuple[int, ...] = (512, 256, 128),
    sample_rate: float = 16384.0,
):
    """
    Create a fast STFT loss function.

    Uses multi-resolution STFT loss with:
    - Spectral convergence loss (70% weight)
    - Log-magnitude loss (30% weight)
    - Zero-crossing rate loss (5% weight)

    Usage:
        loss_fn = combined_loss(target_audio, stft_weight=1.0)
        losses = loss_fn(generated_audio)

    Args:
        target: Target audio [num_samples]
        stft_weight: Weight for STFT loss (default: 1.0)
        stft_fft_sizes: FFT sizes for STFT loss (default: [1024, 512, 256])
        stft_hop_sizes: Hop lengths for STFT loss (default: [512, 256, 128])
        sample_rate: Sample rate (default: 16384.0)

    Returns:
        A function: fn(generated: np.ndarray) -> np.ndarray
        Takes generated audio [batch_size, num_samples] and returns loss [batch_size]
    """
    stft_loss_fn = fast_audio_loss(
        target, stft_fft_sizes, stft_hop_sizes, sample_rate
    )

    if stft_weight == 1.0:
        return stft_loss_fn
    else:
        def loss_fn(generated: np.ndarray) -> np.ndarray:
            """STFT loss with weighting."""
            return stft_weight * stft_loss_fn(generated)
        return loss_fn
