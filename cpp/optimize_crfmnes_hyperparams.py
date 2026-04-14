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
        description="Gaussian-process hyperparameter optimization for CR-FM-NES on compare_optimizers."
    )
    parser.add_argument("--audio", type=Path, default=Path("trumpet-crop.wav"))
    parser.add_argument("--binary", type=Path, default=Path("build/compare_optimizers"))
    parser.add_argument("--calls", type=int, default=14)
    parser.add_argument("--initial-points", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/optimizer_compare/crfmnes_gp_search.json"),
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
        "crfmnes",
        "--json",
        "--crfmnes-population",
        str(params["population_size"]),
        "--crfmnes-sigma",
        str(params["initial_sigma"]),
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
        Integer(16, 96, name="population_size"),
        Real(0.35, 1.25, name="initial_sigma"),
    ]

    history: list[dict] = []

    def objective(values: list[float | int]) -> float:
        params = {
            dim.name: int(value) if isinstance(dim, Integer) else float(value)
            for dim, value in zip(space, values, strict=True)
        }
        if params["population_size"] % 2 != 0:
            params["population_size"] += 1

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
        noise=1e-4,
    )

    best_params = {
        dim.name: int(value) if isinstance(dim, Integer) else float(value)
        for dim, value in zip(space, result.x, strict=True)
    }
    if best_params["population_size"] % 2 != 0:
        best_params["population_size"] += 1

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
