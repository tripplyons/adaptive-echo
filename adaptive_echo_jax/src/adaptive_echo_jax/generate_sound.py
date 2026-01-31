"""
Generate sound by optimizing synthesizer parameters using Differential Evolution on STFT similarity.
"""

import argparse
import os
import time as time_module
from typing import Optional

# Set XLA_FLAGS for multiple CPU devices before importing JAX
# This must be done before JAX is imported to take effect
if "XLA_FLAGS" not in os.environ:
    # Default to 8 CPU devices for pmap parallelism
    # Users can override by setting XLA_FLAGS environment variable
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as np
import numpy as numpy_lib
import soundfile as sf
from scipy import signal  # Keep scipy for resampling (not available in jax.scipy)

from adaptive_echo_jax.constants import (
    num_samples,
    num_seconds,
    training_sample_rate,
    output_sample_rate,
)
from adaptive_echo_jax.loss import combined_loss
from adaptive_echo_jax.synth import synth_parallel


def load_target_audio(
    audio_path: str,
    target_length: int = num_samples,
) -> np.ndarray:
    """Load and preprocess target audio from file."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading target audio from: {audio_path}")
    audio, sr = sf.read(audio_path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != training_sample_rate:
        print(f"Resampling from {sr} Hz to {training_sample_rate} Hz")
        num_samples_old = len(audio)
        num_samples_new = int(num_samples_old * training_sample_rate / sr)
        audio = signal.resample(audio, num_samples_new)

    if audio.shape[0] > target_length:
        audio = audio[:target_length]
    elif audio.shape[0] < target_length:
        padding = target_length - audio.shape[0]
        audio = np.pad(audio, (0, padding))

    audio = audio - audio.mean()
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val

    return audio[np.newaxis, :]


@jax.jit
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


@jax.jit(static_argnames=['population_size', 'num_settings', 'num_trials_per_parent', 'total_trials'])
def de_mutation_crossover(
    population: np.ndarray,
    best_individual: np.ndarray,
    key: jax.Array,
    current_F: float,
    current_Cr: float,
    population_size: int,
    num_settings: int,
    num_trials_per_parent: int,
    total_trials: int,
) -> tuple[np.ndarray, jax.Array]:
    """
    DE/best/1 mutation and binomial crossover.

    Mutation: donor = base + F * (r1 - r2)
    - base is global best 70% of the time, random parent 30% of the time
    - F is dithered randomly [F * 0.5, F * 1.5]
    - Small Gaussian jitter added (decaying with F)

    Crossover: Binomial with rate Cr
    - At least one dimension is guaranteed to come from donor
    """
    # Expand population for all trials
    # Create indices for repeat: [0,0,0,...,1,1,1,...,2,2,2,...]
    indices = np.arange(total_trials) // num_trials_per_parent
    expanded_pop = population[indices]

    # Split random keys
    key, use_best_key, r0_key, r1_key, r2_key, f_dither_key, jitter_key, mask_key, j_rand_key = jax.random.split(key, 9)

    # DE mutation: base is global best 70% of time, random parent 30% of time
    use_best = jax.random.uniform(use_best_key, (total_trials, 1)) < 0.7
    r0 = jax.random.randint(r0_key, (total_trials,), 0, population_size)
    base = np.where(use_best, best_individual[np.newaxis, :], population[r0])

    # Pick random agents for difference
    r1 = jax.random.randint(r1_key, (total_trials,), 0, population_size)
    r2 = jax.random.randint(r2_key, (total_trials,), 0, population_size)

    # Dither F for each trial [current_F * 0.5, current_F * 1.5]
    f_dither = (jax.random.uniform(f_dither_key, (total_trials, 1)) + 0.5) * current_F

    # Generate donor vectors
    donors = base + f_dither * (population[r1] - population[r2])

    # Add small Gaussian jitter for extra exploration (decaying with F)
    donors = donors + jax.random.normal(jitter_key, donors.shape) * (0.1 * current_F)

    # Binomial crossover
    mask = jax.random.uniform(mask_key, (total_trials, num_settings)) < current_Cr

    # Ensure at least one dimension comes from donor
    j_rand = jax.random.randint(j_rand_key, (total_trials,), 0, num_settings)
    mask = mask.at[np.arange(total_trials), j_rand].set(True)

    # Trial vectors (mix donor and parent)
    trials = np.where(mask, donors, expanded_pop)

    return trials, key


@jax.jit(static_argnames=['population_size', 'num_trials_per_parent'])
def de_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    trials: np.ndarray,
    trial_fitness: np.ndarray,
    population_size: int,
    num_trials_per_parent: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DE selection: Select best trial per parent, replace if better.
    """
    # Reshape trial fitness to [population_size, num_trials_per_parent]
    trial_fitness_reshaped = trial_fitness.reshape(population_size, num_trials_per_parent)

    # Find best trial for each parent
    best_trial_local_idx = np.argmin(trial_fitness_reshaped, axis=1)
    best_trial_fitness = trial_fitness_reshaped[np.arange(population_size), best_trial_local_idx]

    # Get global indices of best trials
    global_best_trial_indices = (
        np.arange(population_size) * num_trials_per_parent + best_trial_local_idx
    )
    best_trials_per_parent = trials[global_best_trial_indices]

    # Replace parent if trial is better
    improved = best_trial_fitness < fitness
    new_population = np.where(improved[:, np.newaxis], best_trials_per_parent, population)
    new_fitness = np.where(improved, best_trial_fitness, fitness)

    return new_population, new_fitness


def run_differential_evolution(
    target_audio: np.ndarray,
    time: np.ndarray,
    loss_fn=None,
    stft_weight: float = 1.0,
    population_size: int = 128,
    num_iterations: int = 100,
    F_scale_start: float = 0.8,
    F_scale_end: float = 0.1,
    crossover_rate_start: float = 0.8,
    crossover_rate_end: float = 0.2,
    num_trials_per_parent: int = 12,
    time_limit: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Optimize synthesizer parameters using High-Pressure Differential Evolution.
    Uses DE/best/1 strategy with adaptive scale and crossover decay.

    Features:
    - DE/best/1 mutation: donor = base + F * (r1 - r2)
    - Base is global best 70% of time (exploitation), random 30% of time (exploration)
    - Adaptive F-scale decay: 0.8 → 0.1
    - Adaptive crossover rate decay: 0.8 → 0.2
    - Multiple trials per parent (12-64 per iteration)
    - F-scale dithering [0.5F, 1.5F] for robustness
    - Gaussian jitter proportional to F
    - Diversity injection: Replace worst 5% every 50 iterations

    Args:
        target_audio: Target audio [1, num_samples] or [num_samples]
        time: Time array [num_samples]
        loss_fn: Optional precomputed loss function. If None, creates STFT loss.
        stft_weight: Weight for STFT loss (default: 1.0, used if loss_fn is None)
        population_size: DE population size (default: 128)
        num_iterations: Maximum number of iterations (default: 100)
        F_scale_start: Initial differential weight (default: 0.8)
        F_scale_end: Final differential weight (default: 0.1)
        crossover_rate_start: Initial crossover rate (default: 0.8)
        crossover_rate_end: Final crossover rate (default: 0.2)
        num_trials_per_parent: Trials per parent per iteration (default: 12)
        time_limit: Maximum time in seconds (None for no limit)

    Returns:
        Tuple of (best_settings [1, num_settings], best_loss: float)
    """
    num_settings = 46  # Synth settings size
    t_start_all = time_module.time()

    # Flatten target audio for evaluation
    if target_audio.ndim > 1 and target_audio.shape[0] == 1:
        target_audio_flat = target_audio[0]
    else:
        target_audio_flat = target_audio.flatten()

    # Create loss function if not provided
    if loss_fn is None:
        # Create STFT loss function with precomputed target features (computed once)
        print("Precomputing target features (STFT)...")
        loss_fn = combined_loss(
            target_audio_flat,
            stft_weight=stft_weight,
            sample_rate=training_sample_rate,
        )

    # Initialize population in logit space with wide distribution
    key = jax.random.PRNGKey(42)
    population = jax.random.normal(key, (population_size, num_settings)) * 3.0

    # Evaluate initial population
    settings = sigmoid(population)

    # Synthesize and evaluate in batches to avoid memory issues
    generated_audio = synth_parallel(settings, time)
    fitness = loss_fn(generated_audio)
    fitness.block_until_ready()

    best_idx = np.argmin(fitness)
    best_loss = float(fitness[best_idx])
    best_individual = population[best_idx]

    try:
        for it in range(num_iterations):
            t_it_start = time_module.time()

            # Check time limit
            if (
                time_limit is not None
                and (time_module.time() - t_start_all) > time_limit
            ):
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

            # Generate trial vectors via DE mutation and crossover
            total_trials = population_size * num_trials_per_parent
            key, mut_key = jax.random.split(key)
            trials, key = de_mutation_crossover(
                population,
                best_individual,
                mut_key,
                current_F,
                current_Cr,
                population_size,
                num_settings,
                num_trials_per_parent,
                total_trials,
            )

            # Evaluate all trials
            trial_settings = sigmoid(trials)
            trial_audio = synth_parallel(trial_settings, time)
            trial_fitness = loss_fn(trial_audio)
            trial_fitness.block_until_ready()

            # Selection: Replace parent if best trial is better
            population, fitness = de_selection(
                population,
                fitness,
                trials,
                trial_fitness,
                population_size,
                num_trials_per_parent,
            )

            # Update global best
            current_min_idx = np.argmin(fitness)
            current_loss = float(fitness[current_min_idx])
            if current_loss < best_loss:
                best_loss = current_loss
                best_individual = population[current_min_idx]

            # Diversity Injection (Random Immigrants)
            # Replace the worst 5% of the population with fresh random samples every 50 iterations
            if it > 0 and it % 50 == 0:
                num_immigrants = max(1, population_size // 20)
                worst_idxs = np.argsort(fitness)[-num_immigrants:]

                key, imm_key = jax.random.split(key)
                immigrants = jax.random.normal(imm_key, (num_immigrants, num_settings)) * 3.0

                # Replace worst individuals
                population = population.at[worst_idxs].set(immigrants)

                # Evaluate immigrants
                imm_settings = sigmoid(immigrants)
                imm_audio = synth_parallel(imm_settings, time)
                imm_fitness = loss_fn(imm_audio)
                imm_fitness.block_until_ready()

                fitness = fitness.at[worst_idxs].set(imm_fitness)

            t_it = time_module.time() - t_it_start
            if it % 5 == 0:
                print(
                    f"Iter {it}: Best Loss = {best_loss:.4f}, Time = {t_it:.3f}s, F={current_F:.2f}, Cr={current_Cr:.2f}"
                )

    except KeyboardInterrupt:
        print(
            "\nOptimization interrupted by user. Returning best individual found so far..."
        )

    return sigmoid(best_individual)[np.newaxis, :], best_loss


def main():
    parser = argparse.ArgumentParser(
        description="Optimize synthesizer parameters using Differential Evolution and audio loss functions.",
        epilog="""
Examples:
  # Basic usage (defaults to 8 CPU devices for pmap):
  python -m adaptive_echo_numpy.generate_sound_de input.wav

  # Override number of CPU devices:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" python -m adaptive_echo_numpy.generate_sound_de input.wav

Note: By default, 8 CPU devices are configured for pmap parallelism. You can override
this by setting XLA_FLAGS before running (must be set before JAX is imported).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target", nargs="?", type=str, default=None, help="Path to target audio file"
    )
    parser.add_argument(
        "--population", type=int, default=64, help="Population size (default: 64)"
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of iterations (default: 100)"
    )
    parser.add_argument(
        "--trials", type=int, default=64,
        help="Number of trials per parent (default: 64)"
    )
    parser.add_argument(
        "--stft-weight", type=float, default=1.0,
        help="Weight for fast STFT loss (default: 1.0)"
    )
    args = parser.parse_args()

    time_train = np.linspace(0, num_seconds, num_samples)

    if args.target:
        target_audio = load_target_audio(args.target)
    else:
        target_audio = np.sin(2 * np.pi * time_train * 440)[np.newaxis, :]

    print(f"Running Differential Evolution optimization with STFT loss (weight={args.stft_weight})...")
    optimized_settings, best_loss = run_differential_evolution(
        target_audio,
        time_train,
        stft_weight=args.stft_weight,
        population_size=args.population,
        num_iterations=args.iterations,
        num_trials_per_parent=args.trials,
        time_limit=None,
    )

    print(f"\nFinal best loss: {best_loss:.4f}")

    # Save results (convert JAX arrays to numpy for soundfile)
    # Use output_sample_rate (48000 Hz) for final audio file - higher quality output
    eval_time = np.linspace(0, num_seconds, int(num_seconds * output_sample_rate))
    eval_audio = synth_parallel(optimized_settings, eval_time)
    sf.write("eval_audio_de.wav", numpy_lib.asarray(eval_audio[0]), output_sample_rate)
    print("Saved: eval_audio_de.wav")


if __name__ == "__main__":
    main()
