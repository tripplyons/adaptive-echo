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
from pathlib import Path

from skopt import gp_minimize
from skopt.space import Integer, Real


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gaussian-process coarse-stage optimization for CR-FM-NES on compare_optimizers."
    )
    parser.add_argument("--audio", type=Path, default=Path("trumpet-crop.wav"))
    parser.add_argument("--binary", type=Path, default=Path("build/compare_optimizers"))
    parser.add_argument("--calls", type=int, default=14)
    parser.add_argument("--initial-points", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--population", type=int, default=22)
    parser.add_argument("--sigma", type=float, default=0.378)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/optimizer_compare/crfmnes_coarse_gp_search.json"),
    )
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> None:
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not args.binary.exists():
        raise FileNotFoundError(
            f"Binary not found: {args.binary}. Build with: cmake --build build --config Release --target compare_optimizers --parallel 4"
        )


def run_trial(binary: Path, audio: Path, time_limit: float, population: int, sigma: float,
              params: dict[str, float | int]) -> dict:
    cmd = [
        str(binary),
        str(audio),
        str(time_limit),
        "--optimizer", "crfmnes",
        "--json",
        "--crfmnes-population", str(population),
        "--crfmnes-sigma", str(sigma),
        "--coarse-multiplier", str(params["candidate_multiplier"]),
        "--coarse-min-candidates", str(params["min_candidates"]),
        "--coarse-wide-noise", str(params["wide_noise_std"]),
        "--coarse-medium-noise", str(params["medium_noise_std"]),
        "--coarse-summary-mix", str(params["summary_default_mix"]),
        "--coarse-uniform-mix", str(params["exploratory_uniform_mix"]),
        "--coarse-summary-seed-mix", str(params["exploratory_summary_mix"]),
        "--coarse-default-seed-mix", str(params["exploratory_default_mix"]),
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
        Integer(2, 8, name="candidate_multiplier"),
        Integer(12, 64, name="min_candidates"),
        Real(0.15, 0.55, name="wide_noise_std"),
        Real(0.05, 0.30, name="medium_noise_std"),
        Real(0.35, 0.85, name="summary_default_mix"),
        Real(0.15, 0.70, name="exploratory_uniform_mix"),
        Real(0.10, 0.65, name="exploratory_summary_mix"),
    ]

    history: list[dict] = []

    def objective(values: list[float | int]) -> float:
        params = {
            dim.name: int(value) if isinstance(dim, Integer) else float(value)
            for dim, value in zip(space, values, strict=True)
        }
        remaining = max(0.05, 1.0 - params["exploratory_uniform_mix"] - params["exploratory_summary_mix"])
        params["exploratory_default_mix"] = remaining
        trial = run_trial(args.binary, args.audio, args.time_limit, args.population, args.sigma, params)
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
        noise=1e-4,
    )

    best_params = {
        dim.name: int(value) if isinstance(dim, Integer) else float(value)
        for dim, value in zip(space, result.x, strict=True)
    }
    best_params["exploratory_default_mix"] = max(
        0.05, 1.0 - best_params["exploratory_uniform_mix"] - best_params["exploratory_summary_mix"]
    )

    best_trial = run_trial(args.binary, args.audio, args.time_limit, args.population, args.sigma, best_params)
    payload = {
        "audio": str(args.audio),
        "binary": str(args.binary),
        "time_limit": args.time_limit,
        "population": args.population,
        "sigma": args.sigma,
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
