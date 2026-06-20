from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_dataset.py"


SUITES = {
    "conference_scale": [
        {
            "name": "darcy_structured_realistic_64_medium",
            "grid_size": 64,
            "train_size": 1024,
            "val_size": 160,
            "test_size": 160,
            "task": "darcy_structured_realistic",
            "forcing_scale": 1.0,
            "observation_noise_std": 0.03,
            "seed": 13,
            "output": PROJECT_ROOT / "data" / "darcy_structured_realistic_64_medium.npz",
        },
        {
            "name": "darcy_structured_realistic_80_medium",
            "grid_size": 80,
            "train_size": 640,
            "val_size": 128,
            "test_size": 128,
            "task": "darcy_structured_realistic",
            "forcing_scale": 1.0,
            "observation_noise_std": 0.03,
            "seed": 17,
            "output": PROJECT_ROOT / "data" / "darcy_structured_realistic_80_medium.npz",
        },
        {
            "name": "darcy_structured_realistic_96_small",
            "grid_size": 96,
            "train_size": 384,
            "val_size": 96,
            "test_size": 96,
            "task": "darcy_structured_realistic",
            "forcing_scale": 1.0,
            "observation_noise_std": 0.04,
            "seed": 23,
            "output": PROJECT_ROOT / "data" / "darcy_structured_realistic_96_small.npz",
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a benchmark suite of larger Darcy datasets.")
    parser.add_argument("--suite", type=str, default="conference_scale", choices=sorted(SUITES))
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of dataset names to generate.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--manifest-prefix",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmark_suite_conference_scale",
    )
    return parser.parse_args()


def run_dataset_job(spec: dict, skip_existing: bool) -> Path:
    output_path = Path(spec["output"])
    if skip_existing and output_path.exists():
        return output_path

    command = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--grid-size",
        str(spec["grid_size"]),
        "--train-size",
        str(spec["train_size"]),
        "--val-size",
        str(spec["val_size"]),
        "--test-size",
        str(spec["test_size"]),
        "--task",
        str(spec["task"]),
        "--forcing-scale",
        str(spec["forcing_scale"]),
        "--observation-noise-std",
        str(spec["observation_noise_std"]),
        "--seed",
        str(spec["seed"]),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def summarize_dataset(name: str, dataset_path: Path) -> dict[str, object]:
    with np.load(dataset_path) as data:
        train_coefficients = data["train_coefficients"]
        train_solutions = data["train_solutions"]
        train_clean = data["train_clean_solutions"]
        train_forcings = data["train_forcings"]
        train_boundaries = data["train_boundaries"]
        noise_field = train_solutions - train_clean
        return {
            "name": name,
            "dataset_path": str(dataset_path),
            "grid_size": int(data["grid_size"]),
            "task": str(data["task"]),
            "train_size": int(train_coefficients.shape[0]),
            "val_size": int(data["val_coefficients"].shape[0]),
            "test_size": int(data["test_coefficients"].shape[0]),
            "observation_noise_std": float(data["observation_noise_std"]),
            "coefficient_mean": float(train_coefficients.mean()),
            "coefficient_std": float(train_coefficients.std()),
            "solution_std": float(train_clean.std()),
            "forcing_std": float(train_forcings.std()),
            "boundary_abs_max": float(np.abs(train_boundaries).max()),
            "measured_noise_std": float(noise_field[:, 1:-1, 1:-1].std()),
        }


def write_manifest(prefix: Path, suite_name: str, rows: list[dict[str, object]]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps({"suite": suite_name, "datasets": rows}, indent=2),
        encoding="utf-8",
    )

    lines = [f"# Benchmark Suite: {suite_name}", ""]
    lines.append(
        "| Dataset | Grid | Train/Val/Test | Task | Noise | Coeff Std | Solution Std | Forcing Std | Boundary Max |"
    )
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['grid_size']} | "
            f"{row['train_size']}/{row['val_size']}/{row['test_size']} | "
            f"`{row['task']}` | {row['observation_noise_std']:.3f} | "
            f"{row['coefficient_std']:.4f} | {row['solution_std']:.4f} | "
            f"{row['forcing_std']:.4f} | {row['boundary_abs_max']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    args = parse_args()
    selected_names = set(args.only) if args.only else None
    specs = [
        spec
        for spec in SUITES[args.suite]
        if selected_names is None or spec["name"] in selected_names
    ]
    if not specs:
        raise ValueError("No dataset specifications selected.")

    summaries = []
    for spec in specs:
        dataset_path = run_dataset_job(spec=spec, skip_existing=args.skip_existing)
        summaries.append(summarize_dataset(spec["name"], dataset_path))

    write_manifest(prefix=args.manifest_prefix, suite_name=args.suite, rows=summaries)


if __name__ == "__main__":
    main()
