#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "scikit-optimize>=0.10.2",
# ]
# ///

import argparse
import json
import subprocess
import sys
from pathlib import Path

from skopt import gp_minimize
from skopt.space import Integer, Real


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gaussian-process hyperparameter optimization for Bayes-TPE on compare_optimizers."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("input.wav"),
        help="Input wav file to optimize against.",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build/compare_optimizers"),
        help="Path to compare_optimizers binary.",
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=20,
        help="Total GP optimization calls.",
    )
    parser.add_argument(
        "--initial-points",
        type=int,
        default=6,
        help="Random initial evaluations before GP guidance.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Per-run time limit passed to compare_optimizers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/optimizer_compare/bayes_tpe_gp_search.json"),
        help="Output JSON for best result and search history.",
    )
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> None:
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not args.binary.exists():
        raise FileNotFoundError(
            f"Binary not found: {args.binary}. Build with: cmake --build build --config Release --target compare_optimizers --parallel 4"
        )


def run_trial(binary: Path, audio: Path, time_limit: float, params: dict[str, float | int]) -> dict:
    cmd = [
        str(binary),
        str(audio),
        str(time_limit),
        "--optimizer",
        "bayes-tpe",
        "--json",
        "--tpe-gamma",
        str(params["gamma"]),
        "--tpe-latent-divisor",
        str(params["latent_divisor"]),
        "--tpe-max-latent",
        str(params["max_latent_dim"]),
        "--tpe-min-latent",
        str(params["min_latent_dim"]),
        "--tpe-min-init",
        str(params["min_init_samples"]),
        "--tpe-init-multiplier",
        str(params["init_samples_multiplier"]),
        "--tpe-candidates",
        str(params["candidate_count"]),
        "--tpe-noise-std",
        str(params["local_noise_std"]),
        "--tpe-coarse-radius",
        str(params["coarse_radius"]),
        "--tpe-shape-radius",
        str(params["shape_radius"]),
        "--tpe-refine-radius",
        str(params["refine_radius"]),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"compare_optimizers failed with code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def main() -> int:
    args = parse_args()
    ensure_paths(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    space = [
        Real(0.12, 0.35, name="gamma"),
        Integer(2, 5, name="latent_divisor"),
        Integer(5, 10, name="max_latent_dim"),
        Integer(2, 4, name="min_latent_dim"),
        Integer(8, 18, name="min_init_samples"),
        Integer(3, 6, name="init_samples_multiplier"),
        Integer(192, 640, name="candidate_count"),
        Real(0.05, 0.22, name="local_noise_std"),
        Real(0.12, 0.35, name="coarse_radius"),
        Real(0.08, 0.24, name="shape_radius"),
        Real(0.04, 0.14, name="refine_radius"),
    ]

    history: list[dict] = []

    def objective(values: list[float | int]) -> float:
        params = {
            dim.name: int(value) if isinstance(dim, Integer) else float(value)
            for dim, value in zip(space, values, strict=True)
        }
        if params["min_latent_dim"] > params["max_latent_dim"]:
            params["min_latent_dim"] = params["max_latent_dim"]

        trial = run_trial(args.binary, args.audio, args.time_limit, params)
        history.append(
            {
                "params": params,
                "loss": trial["loss"],
                "elapsed": trial["elapsed"],
                "evals": trial["evals"],
            }
        )
        print(
            f"loss={trial['loss']:.6f} elapsed={trial['elapsed']:.3f}s evals={trial['evals']} params={params}",
            flush=True,
        )
        return float(trial["loss"])

    result = gp_minimize(
        objective,
        space,
        n_calls=args.calls,
        n_initial_points=args.initial_points,
        acq_func="EI",
        random_state=0,
    )

    best_params = {
        dim.name: int(value) if isinstance(dim, Integer) else float(value)
        for dim, value in zip(space, result.x, strict=True)
    }
    if best_params["min_latent_dim"] > best_params["max_latent_dim"]:
        best_params["min_latent_dim"] = best_params["max_latent_dim"]

    best_trial = run_trial(args.binary, args.audio, args.time_limit, best_params)

    payload = {
        "audio": str(args.audio),
        "binary": str(args.binary),
        "time_limit": args.time_limit,
        "calls": args.calls,
        "initial_points": args.initial_points,
        "best_loss": best_trial["loss"],
        "best_elapsed": best_trial["elapsed"],
        "best_evals": best_trial["evals"],
        "best_params": best_params,
        "history": history,
    }

    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nBest loss: {best_trial['loss']:.6f}")
    print(json.dumps(best_params, indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
