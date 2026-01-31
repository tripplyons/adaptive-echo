"""
Generate sound by optimizing synthesizer parameters using Differential Evolution on MRSTFT similarity.
"""

import argparse
import os
import time as time_module
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from adaptive_echo_python.dtw import fast_audio_loss
from adaptive_echo_python.synth import synth_parallel, Synth
from adaptive_echo_python.train import (
    get_device,
    num_samples,
    num_seconds,
    training_sample_rate,
)

# Disable torch.compile to avoid Metal shader buffer limits on MPS
os.environ["TORCH_COMPILE_DISABLE"] = "1"


def load_target_audio(
    audio_path: str,
    device: torch.device,
    target_length: int = num_samples,
) -> torch.Tensor:
    """Load and preprocess target audio from file."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading target audio from: {audio_path}")
    audio, sr = sf.read(str(path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = torch.from_numpy(audio).float()

    if sr != training_sample_rate:
        import torch.nn.functional as F

        print(f"Resampling from {sr} Hz to {training_sample_rate} Hz")
        old_len = audio.shape[0]
        new_len = int(old_len * training_sample_rate / sr)
        audio = (
            F.interpolate(
                audio.unsqueeze(0).unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    if audio.shape[0] > target_length:
        audio = audio[:target_length]
    elif audio.shape[0] < target_length:
        import torch.nn.functional as F

        padding = target_length - audio.shape[0]
        audio = F.pad(audio, (0, padding))

    audio = audio - audio.mean()
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val

    return audio.unsqueeze(0).to(device)


def run_differential_evolution(
    target_audio: torch.Tensor,
    time: torch.Tensor,
    population_size: int = 128,
    num_iterations: int = 100,
    F_scale_start: float = 0.8,
    F_scale_end: float = 0.1,
    crossover_rate_start: float = 0.8,
    crossover_rate_end: float = 0.2,
    num_trials_per_parent: int = 12,
    time_limit: Optional[float] = None,
) -> torch.Tensor:
    """
    Optimize synthesizer parameters using a High-Pressure Differential Evolution.
    Uses DE/best/1 strategy with adaptive scale and crossover decay.
    """
    device = target_audio.device
    num_settings = Synth().encode_settings().shape[0]
    t_start_all = time_module.time()

    # Initialize agents in logit space with wide distribution
    population = torch.randn(population_size, num_settings, device=device) * 3.0

    # Evaluate initial population
    with torch.no_grad():
        settings = torch.sigmoid(population)
        generated_audio = synth_parallel(settings, time)
        fitness = fast_audio_loss(
            generated_audio, target_audio.expand(population_size, -1), verbose=False
        )

    best_idx = torch.argmin(fitness)
    best_loss = fitness[best_idx].item()
    best_individual = population[best_idx].clone()

    try:
        for it in range(num_iterations):
            t_it_start = time_module.time()

            # Check time limit
            if time_limit is not None and (t_it_start - t_start_all) > time_limit:
                print(
                    f"Time limit reached ({time_limit}s). Stopping at iteration {it}."
                )
                break

            # Linear decay for parameters (similar to GA sigma decay)
            progress = it / num_iterations
            current_F = F_scale_start - (F_scale_start - F_scale_end) * progress
            current_Cr = (
                crossover_rate_start
                - (crossover_rate_start - crossover_rate_end) * progress
            )

            with torch.no_grad():
                total_trials = population_size * num_trials_per_parent
                expanded_pop = population.repeat_interleave(
                    num_trials_per_parent, dim=0
                )

                # DE mutation: trials = base + F * (r1 - r2)
                # base is global best 70% of the time, random parent 30% of the time for more exploration
                use_best = torch.rand(total_trials, 1, device=device) < 0.7
                r0 = torch.randint(0, population_size, (total_trials,), device=device)
                base = torch.where(
                    use_best, best_individual.unsqueeze(0), population[r0]
                )

                # Pick random agents for difference
                r1 = torch.randint(0, population_size, (total_trials,), device=device)
                r2 = torch.randint(0, population_size, (total_trials,), device=device)

                # Dither F for each trial to increase robustness [current_F * 0.5, current_F * 1.5]
                f_dither = (
                    torch.rand(total_trials, 1, device=device) + 0.5
                ) * current_F

                # Generate donor vectors
                donors = base + f_dither * (population[r1] - population[r2])

                # Add small Gaussian jitter for extra exploration (decaying with F)
                donors += torch.randn_like(donors) * (0.1 * current_F)

                # 2. Crossover (Uniform masking)
                mask = (
                    torch.rand(total_trials, num_settings, device=device) < current_Cr
                )
                # Ensure at least one dimension is moved
                j_rand = torch.randint(0, num_settings, (total_trials,), device=device)
                mask[torch.arange(total_trials), j_rand] = True

                # Trial vectors (mix donor and parent)
                trials = torch.where(mask, donors, expanded_pop)

                # 3. Evaluation
                trial_settings = torch.sigmoid(trials)
                trial_audio = synth_parallel(trial_settings, time)
                all_trial_fitness = fast_audio_loss(
                    trial_audio, target_audio.expand(total_trials, -1), verbose=False
                )

                # 4. Selection (Local best trial for each parent)
                trial_fitness_reshaped = all_trial_fitness.view(
                    population_size, num_trials_per_parent
                )
                best_trial_fitness, best_trial_local_idx = torch.min(
                    trial_fitness_reshaped, dim=1
                )

                global_best_trial_indices = (
                    torch.arange(population_size, device=device) * num_trials_per_parent
                    + best_trial_local_idx
                )
                best_trials_per_parent = trials[global_best_trial_indices]

                # 5. Replace parent if better
                improved = best_trial_fitness < fitness
                population[improved] = best_trials_per_parent[improved]
                fitness[improved] = best_trial_fitness[improved]

                # 6. Update global best
                current_min_idx = torch.argmin(fitness)
                if fitness[current_min_idx] < best_loss:
                    best_loss = fitness[current_min_idx].item()
                    best_individual = population[current_min_idx].clone()

                # 7. Diversity Injection (Random Immigrants)
                # Replace the worst 5% of the population with fresh random samples every 50 iterations
                if it > 0 and it % 50 == 0:
                    num_immigrants = max(1, population_size // 20)
                    worst_idxs = torch.argsort(fitness, descending=True)[
                        :num_immigrants
                    ]
                    immigrants = (
                        torch.randn(num_immigrants, num_settings, device=device) * 3.0
                    )
                    population[worst_idxs] = immigrants

                    # Evaluate immigrants immediately
                    imm_settings = torch.sigmoid(immigrants)
                    imm_audio = synth_parallel(imm_settings, time)
                    imm_fitness = fast_audio_loss(
                        imm_audio,
                        target_audio.expand(num_immigrants, -1),
                        verbose=False,
                    )
                    fitness[worst_idxs] = imm_fitness

            t_it = time_module.time() - t_it_start
            if it % 5 == 0:
                print(
                    f"Iter {it}: Best Loss = {best_loss:.4f}, Time = {t_it:.3f}s, F={current_F:.2f}, Cr={current_Cr:.2f}"
                )

    except KeyboardInterrupt:
        print(
            "\nOptimization interrupted by user. Returning best individual found so far..."
        )

    return torch.sigmoid(best_individual).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(
        description="Optimize synthesizer parameters using DE and MRSTFT."
    )
    parser.add_argument(
        "target", nargs="?", type=str, default=None, help="Path to target audio file"
    )
    args = parser.parse_args()

    device = get_device()
    time_train = torch.linspace(0, num_seconds, num_samples, device=device)

    if args.target:
        target_audio = load_target_audio(args.target, device)
    else:
        target_audio = torch.sin(2 * torch.pi * time_train * 440).unsqueeze(0)

    print("Running Differential Evolution optimization with Multi-Resolution STFT...")
    optimized_settings = run_differential_evolution(
        target_audio,
        time_train,
        population_size=64,
        num_iterations=100,
        num_trials_per_parent=64,
        time_limit=None,
    )

    # Save results
    eval_time = torch.linspace(0, num_seconds, num_seconds * 48000, device=device)
    with torch.inference_mode():
        eval_audio = synth_parallel(optimized_settings, eval_time)
        sf.write("eval_audio_de.wav", eval_audio.cpu().numpy()[0], 48000)
    print("Saved: eval_audio_de.wav")


if __name__ == "__main__":
    main()
