from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import time as time_module
from tslearn.metrics import SoftDTWLossPyTorch, dtw, cdist_dtw

import torch.nn.functional as F


def mrstft_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    fft_sizes=[2048, 1024, 512, 256, 64],
    hop_sizes=[512, 256, 128, 64, 16],
    win_lengths=[2048, 1024, 512, 256, 64],
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute Multi-Resolution STFT loss.
    Returns a tensor of shape [batch_size] containing the loss for each sample.
    """
    t_start = time_module.time()
    batch_size = generated.shape[0]
    total_loss = torch.zeros(batch_size, device=generated.device)

    # Normalization (optional but often helpful for similarity)
    gen_norm = generated - generated.mean(dim=-1, keepdim=True)
    gen_norm = gen_norm / (gen_norm.std(dim=-1, keepdim=True) + 1e-8)
    tgt_norm = target - target.mean(dim=-1, keepdim=True)
    tgt_norm = tgt_norm / (tgt_norm.std(dim=-1, keepdim=True) + 1e-8)

    for n_fft, hop_length, win_length in zip(fft_sizes, hop_sizes, win_lengths):
        window = torch.hann_window(win_length).to(generated.device)

        # STFT for generated
        x_stft = torch.stft(
            gen_norm,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        x_mag = torch.abs(x_stft)

        # STFT for target
        y_stft = torch.stft(
            tgt_norm,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        y_mag = torch.abs(y_stft)

        # Spectral Convergence Loss
        # x_mag shape: [batch, freq, time]
        sc_loss = torch.linalg.matrix_norm(y_mag - x_mag, ord="fro") / (
            torch.linalg.matrix_norm(y_mag, ord="fro") + 1e-8
        )

        # Log-Magnitude Loss
        mag_loss = torch.mean(
            torch.abs(torch.log(y_mag + 1e-7) - torch.log(x_mag + 1e-7)), dim=(1, 2)
        )

        total_loss += sc_loss + mag_loss

    total_loss /= len(fft_sizes)

    t_total = time_module.time() - t_start
    if verbose:
        print(f"    [MRSTFT] scales:{len(fft_sizes)}, total:{t_total:.3f}s")

    return total_loss


def fast_audio_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    fft_sizes=[1024, 512, 256],
    hop_sizes=[512, 256, 128],
    win_lengths=[1024, 512, 256],
    verbose: bool = True,
) -> torch.Tensor:
    """
    Fast audio similarity loss optimized for speed.
    Uses reduced sample rate (8192 Hz) and fewer/smaller FFT sizes.

    Args:
        generated: Generated audio tensor [batch_size, num_samples]
        target: Target audio tensor [batch_size, num_samples] or [num_samples]
        sample_rate: Target sample rate for processing (default 8192, minimum 8192)
        fft_sizes: List of FFT window sizes (default [512, 256, 128])
        hop_sizes: List of hop sizes corresponding to fft_sizes
        win_lengths: List of window lengths corresponding to fft_sizes
        verbose: Whether to print timing information

    Returns:
        Tensor of shape [batch_size] containing loss for each sample
    """
    t_start = time_module.time()
    device = generated.device
    batch_size = generated.shape[0]
    
    gen_resampled = generated
    if target.dim() == 1:
        tgt_resampled = target.unsqueeze(0).expand(batch_size, -1)
    else:
        tgt_resampled = target

    # Fast normalization (in-place operations where possible)
    gen_mean = gen_resampled.mean(dim=-1, keepdim=True)
    gen_norm = gen_resampled - gen_mean
    gen_std = gen_norm.std(dim=-1, keepdim=True)
    gen_norm = gen_norm / (gen_std + 1e-8)

    tgt_mean = tgt_resampled.mean(dim=-1, keepdim=True)
    tgt_norm = tgt_resampled - tgt_mean
    tgt_std = tgt_norm.std(dim=-1, keepdim=True)
    tgt_norm = tgt_norm / (tgt_std + 1e-8)

    total_loss = torch.zeros(batch_size, device=device)

    # Compute STFT losses with smaller FFT sizes
    for n_fft, hop_length, win_length in zip(fft_sizes, hop_sizes, win_lengths):
        # Use cached Hann window
        window = torch.hann_window(win_length, device=device)

        # STFT for generated
        x_stft = torch.stft(
            gen_norm,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True,
        )
        x_mag = torch.abs(x_stft)

        # STFT for target
        y_stft = torch.stft(
            tgt_norm,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True,
        )
        y_mag = torch.abs(y_stft)

        # Fast spectral convergence using mean instead of Frobenius norm
        # This is faster and still captures magnitude differences
        sc_loss = torch.mean(
            torch.abs(y_mag - x_mag) / (y_mag.mean(dim=(1, 2), keepdim=True) + 1e-8),
            dim=(1, 2),
        )

        # Log-magnitude loss with smaller epsilon for numerical stability
        log_y = torch.log(y_mag + 1e-8)
        log_x = torch.log(x_mag + 1e-8)
        mag_loss = torch.mean(torch.abs(log_y - log_x), dim=(1, 2))

        # Weight the losses (spectral convergence gets higher weight for timbre)
        total_loss += 0.7 * sc_loss + 0.3 * mag_loss

    # Average over scales
    total_loss /= len(fft_sizes)

    # Add envelope correlation loss for temporal similarity (very fast)
    # Simple zero-crossing rate difference as a proxy for temporal structure
    gen_zcr = ((gen_norm[:, 1:] * gen_norm[:, :-1]) < 0).float().mean(dim=-1)
    tgt_zcr = ((tgt_norm[:, 1:] * tgt_norm[:, :-1]) < 0).float().mean(dim=-1)
    zcr_loss = torch.abs(gen_zcr - tgt_zcr)

    # Combine with small weight
    total_loss = 0.95 * total_loss + 0.05 * zcr_loss

    t_total = time_module.time() - t_start
    if verbose:
        print(
            f"    [FastLoss] sr:{sample_rate}, scales:{len(fft_sizes)}, total:{t_total:.3f}s"
        )

    return total_loss


def soft_dtw_loss_windowed(
    generated: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 1.0,
    window_size: int = 64,
    hop_length: int = 32,
    verbose: bool = True,
) -> torch.Tensor:
    """Compute soft DTW loss on overlapping windows using tslearn's SoftDTWLossPyTorch."""
    t_start = time_module.time()
    batch_size, seq_len = generated.shape

    t_norm = time_module.time()
    gen_norm = generated - generated.mean(dim=-1, keepdim=True)
    gen_norm = gen_norm / (gen_norm.std(dim=-1, keepdim=True) + 1e-8)
    tgt_norm = target - target.mean(dim=-1, keepdim=True)
    tgt_norm = tgt_norm / (tgt_norm.std(dim=-1, keepdim=True) + 1e-8)
    t_norm = time_module.time() - t_norm

    t_unfold = time_module.time()
    # Unfold into windows: [batch_size, num_windows, window_size]
    gen_windows = gen_norm.unfold(dimension=1, size=window_size, step=hop_length)
    tgt_windows = tgt_norm.unfold(dimension=1, size=window_size, step=hop_length)
    num_windows = gen_windows.shape[1]
    t_unfold = time_module.time() - t_unfold

    t_dtw = time_module.time()
    # Reshape for tslearn: [n_samples, sz, d]
    # We treat all windows in the batch as separate samples
    x = gen_windows.reshape(-1, window_size, 1)
    y = tgt_windows.reshape(-1, window_size, 1)

    # Use tslearn's SoftDTWLossPyTorch with Euclidean distance (p=2) to match original behavior.
    criterion = SoftDTWLossPyTorch(
        gamma=gamma, dist_func=lambda x, y: torch.cdist(x, y, p=2)
    )
    loss_values = criterion(x, y)
    result = loss_values.mean()
    t_dtw = time_module.time() - t_dtw

    t_total = time_module.time() - t_start
    if verbose:
        print(
            f"    [DTW] windows:{num_windows}, norm:{t_norm:.3f}s, unfold:{t_unfold:.3f}s, "
            f"dtw:{t_dtw:.3f}s, total:{t_total:.3f}s"
        )

    return result


def sakoe_chiba_dtw_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    sakoe_chiba_radius: int = 256,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute non-differentiable DTW loss with Sakoe-Chiba global constraint on full waveforms.
    Intended for use with Genetic Algorithms.
    """
    t_start = time_module.time()
    batch_size, seq_len = generated.shape

    t_norm_start = time_module.time()
    # Normalization
    gen_norm = generated - generated.mean(dim=-1, keepdim=True)
    gen_norm = gen_norm / (gen_norm.std(dim=-1, keepdim=True) + 1e-8)
    tgt_norm = target - target.mean(dim=-1, keepdim=True)
    tgt_norm = tgt_norm / (tgt_norm.std(dim=-1, keepdim=True) + 1e-8)
    t_norm = time_module.time() - t_norm_start

    t_conv_start = time_module.time()
    # Convert to numpy for tslearn: [n_samples, sz, d]
    gen_np = gen_norm.detach().cpu().numpy()[..., None]
    tgt_np = tgt_norm.detach().cpu().numpy()[..., None]
    t_conv = time_module.time() - t_conv_start

    t_dtw_start = time_module.time()
    # cdist_dtw returns [batch_size, 1]
    costs = cdist_dtw(
        gen_np,
        tgt_np[0:1],
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=sakoe_chiba_radius,
    )
    final_losses = costs.flatten()
    t_dtw = time_module.time() - t_dtw_start

    result = torch.tensor(final_losses, device=generated.device, dtype=generated.dtype)

    t_total = time_module.time() - t_start
    if verbose:
        print(
            f"    [Sakoe-Chiba DTW Full] norm:{t_norm:.3f}s, cpu_conv:{t_conv:.3f}s, dtw:{t_dtw:.3f}s, total:{t_total:.3f}s"
        )

    return result
