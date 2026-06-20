from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


NUMERIC_FIELDS = [
    "test_rmse",
    "test_nll",
    "sharpness_mean_std",
    "uncertainty_error_corr",
    "interval_calibration_mae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment results into JSON and Markdown.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories containing evaluation_metrics.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Experiment Summary")
    return parser.parse_args()


def load_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "evaluation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = {"run": run_dir.name, **metrics}
    return payload


def numeric_stats(runs: list[dict]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for field in NUMERIC_FIELDS:
        values = [run[field] for run in runs if isinstance(run.get(field), (int, float))]
        if not values:
            continue
        stats[field] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return stats


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, title: str, runs: list[dict], stats: dict[str, dict[str, float]]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| Run | RMSE | NLL | Sharpness | Unc-Err Corr | Calib MAE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for run in runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    run["run"],
                    format_value(run.get("test_rmse")),
                    format_value(run.get("test_nll")),
                    format_value(run.get("sharpness_mean_std")),
                    format_value(run.get("uncertainty_error_corr")),
                    format_value(run.get("interval_calibration_mae")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    for field, field_stats in stats.items():
        lines.append(
            f"- `{field}`: mean={field_stats['mean']:.4f}, std={field_stats['std']:.4f}, "
            f"min={field_stats['min']:.4f}, max={field_stats['max']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dirs = [Path(run) for run in args.runs]
    runs = [load_run(run_dir) for run_dir in run_dirs]
    stats = numeric_stats(runs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"title": args.title, "runs": runs, "aggregate": stats}, indent=2),
        encoding="utf-8",
    )
    write_markdown(path=args.output_md, title=args.title, runs=runs, stats=stats)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
