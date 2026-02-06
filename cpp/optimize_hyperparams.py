#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "optuna>=3.0.0",
#     "pandas",
# ]
# ///

"""
Hyperparameter optimization for adaptive_echo using Optuna.

This script uses uv's inline dependency management. Run with:
    uv run optimize_hyperparams.py --audio target.wav --trials 50

Or install dependencies manually:
    pip install optuna
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for adaptive_echo C++ binary"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to target audio file",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=35,
        help="Timeout per trial in seconds (default: 35)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="adaptive_echo_optimization",
        help="Optuna study name (default: adaptive_echo_optimization)",
    )
    return parser.parse_args()


def check_binary_exists():
    """Check if the C++ binary exists."""
    binary_path = Path("./build/generate_sound")
    if not binary_path.exists():
        print("Error: C++ binary not found at ./build/generate_sound")
        print("Please build the project first:")
        print("  mkdir -p build && cd build && cmake .. && make")
        sys.exit(1)
    return binary_path


def run_generate_sound(binary_path, audio_path, population, sigma, shade_memory_size,
                       archive_multiplier, stagnation_threshold, cr_std, f_scale, timeout):
    """
    Run the generate_sound binary with given hyperparameters.

    Returns:
        tuple: (best_loss, output_dict) or (None, None) on failure
    """
    cmd = [
        str(binary_path),
        audio_path,
        "--population", str(population),
        "--sigma", str(sigma),
        "--shade-memory", str(shade_memory_size),
        "--archive-multiplier", str(archive_multiplier),
        "--stagnation", str(stagnation_threshold),
        "--cr-std", str(cr_std),
        "--f-scale", str(f_scale),
        "--time-limit", "30",
        "--json",
    ]
    
    result = None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            print(f"  Warning: Binary exited with code {result.returncode}")
            print(f"  stderr: {result.stderr[:200]}")
            return None, None

        # Parse JSON output
        output = json.loads(result.stdout)
        best_loss = output.get("best_loss")

        if best_loss is None:
            print("  Warning: No best_loss in output")
            return None, None

        return best_loss, output

    except subprocess.TimeoutExpired:
        print(f"  Warning: Trial timed out after {timeout}s")
        return None, None
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON output: {e}")
        if result:
            print(f"  stdout: {repr(result.stdout[:500])}")
            print(f"  stderr: {repr(result.stderr[:500])}")
        return None, None
    except Exception as e:
        print(f"  Warning: Error running binary: {e}")
        return None, None


def objective(trial, binary_path, audio_path, timeout):
    """
    Optuna objective function to minimize best_loss.
    """
    # Suggest hyperparameters
    population_size = trial.suggest_int("population_size", 16, 128)
    initial_sigma = trial.suggest_float("initial_sigma", 0.5, 5.0)
    shade_memory_size = trial.suggest_int("shade_memory_size", 2, 10)
    archive_multiplier = trial.suggest_int("archive_multiplier", 1, 5)
    stagnation_threshold = trial.suggest_int("stagnation_threshold", 10, 100)
    cr_std = trial.suggest_float("cr_std", 0.01, 0.3)
    f_scale = trial.suggest_float("f_scale", 0.01, 0.3)

    print(f"\nTrial {trial.number + 1}: population={population_size}, "
          f"sigma={initial_sigma:.3f}, shade_memory={shade_memory_size}, "
          f"archive_mult={archive_multiplier}, stagnation={stagnation_threshold}, "
          f"cr_std={cr_std:.3f}, f_scale={f_scale:.3f}")

    # Run the binary
    best_loss, output = run_generate_sound(
        binary_path,
        audio_path,
        population_size,
        initial_sigma,
        shade_memory_size,
        archive_multiplier,
        stagnation_threshold,
        cr_std,
        f_scale,
        timeout,
    )
    
    if best_loss is None:
        # Return a large value to indicate failure
        return float("inf")
    
    print(f"  -> best_loss: {best_loss:.6f}")
    
    return best_loss


def main():
    args = parse_args()
    
    # Check if audio file exists
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Check if binary exists
    binary_path = check_binary_exists()
    print(f"Using binary: {binary_path}")
    print(f"Audio file: {args.audio}")
    print(f"Number of trials: {args.trials}")
    print(f"Timeout per trial: {args.timeout}s")
    print(f"Study name: {args.study_name}")
    print("-" * 50)
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),  # Bayesian optimization
    )
    
    # Define objective with fixed arguments
    def objective_wrapper(trial):
        return objective(trial, binary_path, args.audio, args.timeout)
    
    # Run optimization
    try:
        study.optimize(
            objective_wrapper,
            n_trials=args.trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    
    # Print results
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    
    if len(study.trials) == 0:
        print("No trials completed successfully.")
        sys.exit(1)
    
    print(f"\nNumber of trials completed: {len(study.trials_dataframe())}")
    print(f"Number of pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
    print(f"Number of failed trials: {len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))}")
    
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"Best loss: {best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    print(f"  population_size: {best_trial.params['population_size']}")
    print(f"  initial_sigma: {best_trial.params['initial_sigma']:.6f}")
    print(f"  shade_memory_size: {best_trial.params['shade_memory_size']}")
    print(f"  archive_multiplier: {best_trial.params['archive_multiplier']}")
    print(f"  stagnation_threshold: {best_trial.params['stagnation_threshold']}")
    print(f"  cr_std: {best_trial.params['cr_std']:.6f}")
    print(f"  f_scale: {best_trial.params['f_scale']:.6f}")

    # Save best hyperparameters to JSON
    output_file = "best_hyperparams.json"
    best_params = {
        "population_size": best_trial.params["population_size"],
        "initial_sigma": best_trial.params["initial_sigma"],
        "shade_memory_size": best_trial.params["shade_memory_size"],
        "archive_multiplier": best_trial.params["archive_multiplier"],
        "stagnation_threshold": best_trial.params["stagnation_threshold"],
        "cr_std": best_trial.params["cr_std"],
        "f_scale": best_trial.params["f_scale"],
        "best_loss": best_trial.value,
        "study_name": args.study_name,
        "trial_number": best_trial.number,
    }
    
    with open(output_file, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nBest hyperparameters saved to: {output_file}")
    
    # Print usage example
    print("\nTo run with best hyperparameters:")
    print(f"  ./build/generate_sound {args.audio} ", end="")
    print(f"--population {best_trial.params['population_size']} ", end="")
    print(f"--sigma {best_trial.params['initial_sigma']:.6f} ", end="")
    print(f"--shade-memory {best_trial.params['shade_memory_size']} ", end="")
    print(f"--archive-multiplier {best_trial.params['archive_multiplier']} ", end="")
    print(f"--stagnation {best_trial.params['stagnation_threshold']} ", end="")
    print(f"--cr-std {best_trial.params['cr_std']:.6f} ", end="")
    print(f"--f-scale {best_trial.params['f_scale']:.6f}")


if __name__ == "__main__":
    main()
