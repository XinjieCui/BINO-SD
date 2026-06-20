from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bi_operator.pde import apply_interior_observation_noise, generate_darcy_sample


def build_split(
    size: int,
    grid_size: int,
    task: str,
    forcing_scale: float,
    observation_noise_std: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    coefficients = np.zeros((size, grid_size, grid_size), dtype=np.float32)
    solutions = np.zeros((size, grid_size, grid_size), dtype=np.float32)
    clean_solutions = np.zeros((size, grid_size, grid_size), dtype=np.float32)
    forcings = np.zeros((size, grid_size, grid_size), dtype=np.float32)
    boundaries = np.zeros((size, grid_size, grid_size), dtype=np.float32)
    for index in range(size):
        coefficient, clean_solution, forcing, boundary = generate_darcy_sample(
            grid_size=grid_size,
            rng=rng,
            task=task,
            forcing_scale=forcing_scale,
        )
        solution = apply_interior_observation_noise(
            solution=clean_solution,
            boundary=boundary,
            rng=rng,
            relative_std=observation_noise_std,
        )
        coefficients[index] = coefficient
        solutions[index] = solution
        clean_solutions[index] = clean_solution
        forcings[index] = forcing
        boundaries[index] = boundary
    return {
        "coefficients": coefficients,
        "solutions": solutions,
        "clean_solutions": clean_solutions,
        "forcings": forcings,
        "boundaries": boundaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic Darcy inverse dataset.")
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--val-size", type=int, default=48)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument(
        "--task",
        type=str,
        default="darcy_constant",
        choices=[
            "darcy_constant",
            "darcy_multisource",
            "darcy_structured_multisource",
            "darcy_structured_realistic",
        ],
    )
    parser.add_argument("--forcing-scale", type=float, default=1.0)
    parser.add_argument("--observation-noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "darcy_inverse_dataset.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    train_split = build_split(
        size=args.train_size,
        grid_size=args.grid_size,
        task=args.task,
        forcing_scale=args.forcing_scale,
        observation_noise_std=args.observation_noise_std,
        rng=rng,
    )
    val_split = build_split(
        size=args.val_size,
        grid_size=args.grid_size,
        task=args.task,
        forcing_scale=args.forcing_scale,
        observation_noise_std=args.observation_noise_std,
        rng=rng,
    )
    test_split = build_split(
        size=args.test_size,
        grid_size=args.grid_size,
        task=args.task,
        forcing_scale=args.forcing_scale,
        observation_noise_std=args.observation_noise_std,
        rng=rng,
    )

    np.savez_compressed(
        args.output,
        train_coefficients=train_split["coefficients"],
        train_solutions=train_split["solutions"],
        train_clean_solutions=train_split["clean_solutions"],
        train_forcings=train_split["forcings"],
        train_boundaries=train_split["boundaries"],
        val_coefficients=val_split["coefficients"],
        val_solutions=val_split["solutions"],
        val_clean_solutions=val_split["clean_solutions"],
        val_forcings=val_split["forcings"],
        val_boundaries=val_split["boundaries"],
        test_coefficients=test_split["coefficients"],
        test_solutions=test_split["solutions"],
        test_clean_solutions=test_split["clean_solutions"],
        test_forcings=test_split["forcings"],
        test_boundaries=test_split["boundaries"],
        grid_size=np.array(args.grid_size, dtype=np.int64),
        forcing=np.array(args.forcing_scale, dtype=np.float32),
        task=np.array(args.task),
        seed=np.array(args.seed, dtype=np.int64),
        observation_noise_std=np.array(args.observation_noise_std, dtype=np.float32),
    )

    print(f"Dataset written to: {args.output}")
    print(
        "Split sizes:",
        {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
            "grid_size": args.grid_size,
            "task": args.task,
            "observation_noise_std": args.observation_noise_std,
        },
    )


if __name__ == "__main__":
    main()
