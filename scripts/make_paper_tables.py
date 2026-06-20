from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style markdown and LaTeX tables.")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/paper_tables_2026-03-22.md"),
        help="Markdown output path.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("artifacts/paper_tables_2026-03-22.tex"),
        help="LaTeX output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(root: Path, run_name: str) -> dict:
    payload = load_json(root / "artifacts" / run_name / "evaluation_metrics.json")
    payload["run"] = run_name
    return payload


def load_aggregate_row(summary_path: Path, label: str) -> dict:
    payload = load_json(summary_path)
    aggregate = payload["aggregate"]
    return {
        "label": label,
        "rmse_mean": aggregate["test_rmse"]["mean"],
        "rmse_std": aggregate["test_rmse"]["std"],
        "nll_mean": aggregate["test_nll"]["mean"],
        "nll_std": aggregate["test_nll"]["std"],
        "calib_mean": aggregate["interval_calibration_mae"]["mean"],
        "calib_std": aggregate["interval_calibration_mae"]["std"],
    }


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def fmt_mean_std(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None or std_value is None:
        return "-"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def latex_table(caption: str, label: str, headers: list[str], rows: list[list[str]]) -> str:
    column_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{tex_escape(caption)}}}",
        f"\\label{{{tex_escape(label)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        " & ".join(tex_escape(header) for header in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(cell) for cell in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    main_runs = [
        ("Deterministic operator", 16, load_metrics(root, "baseline_det_operator")),
        ("U-Net deterministic", 16, load_metrics(root, "baseline_unet_det")),
        ("Bayesian + fixed grid", 16, load_metrics(root, "baseline_fixed_grid")),
        ("Bayesian + random grid", 16, load_metrics(root, "baseline_random_grid")),
        ("Bayesian + learned G-R", 16, load_metrics(root, "sensor_learned_gaussian_random")),
        ("Bayesian + learned G-R", 24, load_metrics(root, "abl_sensors24")),
    ]
    main_multiseed_16 = load_aggregate_row(root / "artifacts" / "multiseed_summary.json", "Bayesian + learned G-R (16, 3 seeds)")
    main_multiseed_24 = load_aggregate_row(root / "artifacts" / "multiseed_summary_24s.json", "Bayesian + learned G-R (24, 3 seeds)")

    ablation_map = {
        "sensor_learned_gaussian_random": "Base (16 sensors)",
        "abl_no_repulsion": "No repulsion",
        "abl_no_physics": "No physics residual",
        "abl_no_nll": "No NLL term",
        "abl_sensors8": "8 sensors",
        "abl_sensors24": "24 sensors",
        "abl_small_data": "Smaller dataset",
    }
    ablation_payload = load_json(root / "artifacts" / "ablation_summary.json")
    ablation_rows_raw = {run["run"]: run for run in ablation_payload["runs"]}

    task_rows = [
        ("20x20 constant", "16 learned sensors", load_metrics(root, "sensor_learned_gaussian_random")),
        ("20x20 constant", "24 learned sensors", load_metrics(root, "abl_sensors24")),
        ("32x32 constant", "24 learned sensors", load_metrics(root, "task_constant32_lgr")),
        ("32x32 multisource", "24 learned sensors (old)", load_metrics(root, "task_multisource32_lgr")),
        ("32x32 multisource", "40 learned sensors (best)", load_metrics(root, "task_multisource32_big40")),
        ("32x32 multisource", "40 random sensors", load_metrics(root, "task_multisource32_random40")),
        ("32x32 multisource", "40 fixed sensors", load_metrics(root, "task_multisource32_fixed40")),
        ("48x48 structured multisource", "40 learned sensors", load_metrics(root, "task_structured48_big40_v2")),
        ("64x64 structured realistic", "64 learned sensors", load_metrics(root, "task_structured_realistic64_main64s")),
        ("80x80 structured realistic", "56 learned sensors", load_metrics(root, "task_structured_realistic80_main")),
        ("96x96 structured realistic", "72 learned sensors", load_metrics(root, "task_structured_realistic96_main")),
    ]

    hard_task_rows = [
        ("32x32 multisource", "Initial 24-sensor run", load_metrics(root, "task_multisource32_lgr")),
        ("32x32 multisource", "Tuned 32-sensor run", load_metrics(root, "task_multisource32_tuned_32s")),
        ("32x32 multisource", "Learned 40-sensor run", load_metrics(root, "task_multisource32_big40")),
        ("32x32 multisource", "Random 40-sensor run", load_metrics(root, "task_multisource32_random40")),
        ("32x32 multisource", "Fixed 40-sensor run", load_metrics(root, "task_multisource32_fixed40")),
        ("48x48 structured multisource", "First large structured benchmark", load_metrics(root, "task_structured48_big40_v2")),
    ]
    hard_multiseed = load_aggregate_row(
        root / "artifacts" / "multiseed_summary_multisource32_big40.json",
        "32x32 multisource learned40 (3 seeds)",
    )
    hard_random40_multiseed = load_aggregate_row(
        root / "artifacts" / "multiseed_summary_multisource32_random40.json",
        "32x32 multisource random40 (3 seeds)",
    )
    realistic_multiseed = load_aggregate_row(
        root / "artifacts" / "multiseed_summary_realistic64_64s.json",
        "64x64 realistic (64 sensors, 3 seeds)",
    )
    realistic80_multiseed = load_aggregate_row(
        root / "artifacts" / "multiseed_summary_realistic80_56s.json",
        "80x80 realistic (56 sensors, 3 seeds)",
    )
    realistic96_multiseed = load_aggregate_row(
        root / "artifacts" / "multiseed_summary_realistic96_72s.json",
        "96x96 realistic (72 sensors, 3 seeds)",
    )
    structured_rows = [
        ("Bayesian + learned G-R", load_metrics(root, "task_structured48_big40_v2")),
        ("Bayesian + random grid", load_metrics(root, "task_structured48_random40")),
        ("Deterministic operator", load_metrics(root, "task_structured48_det40")),
    ]
    realistic_rows = [
        ("Bayesian + learned G-R", 64, load_metrics(root, "task_structured_realistic64_main64s")),
        ("Bayesian + random grid", 48, load_metrics(root, "task_structured_realistic64_random48")),
        ("Bayesian + fixed grid", 48, load_metrics(root, "task_structured_realistic64_fixed48")),
        ("Deterministic operator", 48, load_metrics(root, "task_structured_realistic64_det48")),
        ("U-Net deterministic", 48, load_metrics(root, "task_structured_realistic64_unet48")),
    ]
    realistic80_rows = [
        ("Bayesian + learned G-R", 56, load_metrics(root, "task_structured_realistic80_main")),
        ("Bayesian + random grid", 56, load_metrics(root, "task_structured_realistic80_random56")),
        ("Bayesian + fixed grid", 56, load_metrics(root, "task_structured_realistic80_fixed56")),
        ("Deterministic operator", 56, load_metrics(root, "task_structured_realistic80_det56")),
        ("U-Net deterministic", 56, load_metrics(root, "task_structured_realistic80_unet56")),
    ]
    realistic96_rows = [
        ("Bayesian + learned G-R", 72, load_metrics(root, "task_structured_realistic96_main")),
        ("Bayesian + random grid", 72, load_metrics(root, "task_structured_realistic96_random72")),
        ("Bayesian + fixed grid", 72, load_metrics(root, "task_structured_realistic96_fixed72")),
        ("Deterministic operator", 72, load_metrics(root, "task_structured_realistic96_det72")),
        ("U-Net deterministic", 72, load_metrics(root, "task_structured_realistic96_unet72")),
    ]

    main_headers = ["Method", "Sensors", "RMSE", "NLL", "Calib MAE"]
    main_rows = [
        [label, str(sensor_count), fmt(metrics.get("test_rmse")), fmt(metrics.get("test_nll")), fmt(metrics.get("interval_calibration_mae"))]
        for label, sensor_count, metrics in main_runs
    ]
    main_rows.extend(
        [
            [
                main_multiseed_16["label"],
                "16",
                fmt_mean_std(main_multiseed_16["rmse_mean"], main_multiseed_16["rmse_std"]),
                fmt_mean_std(main_multiseed_16["nll_mean"], main_multiseed_16["nll_std"]),
                fmt_mean_std(main_multiseed_16["calib_mean"], main_multiseed_16["calib_std"]),
            ],
            [
                main_multiseed_24["label"],
                "24",
                fmt_mean_std(main_multiseed_24["rmse_mean"], main_multiseed_24["rmse_std"]),
                fmt_mean_std(main_multiseed_24["nll_mean"], main_multiseed_24["nll_std"]),
                fmt_mean_std(main_multiseed_24["calib_mean"], main_multiseed_24["calib_std"]),
            ],
        ]
    )

    ablation_headers = ["Ablation", "RMSE", "NLL", "Calib MAE"]
    ablation_rows = [
        [
            ablation_map[run_name],
            fmt(ablation_rows_raw[run_name].get("test_rmse")),
            fmt(ablation_rows_raw[run_name].get("test_nll")),
            fmt(ablation_rows_raw[run_name].get("interval_calibration_mae")),
        ]
        for run_name in ablation_map
    ]

    task_headers = ["Task", "Configuration", "RMSE", "NLL", "Calib MAE"]
    task_rows_formatted = [
        [task_name, config_name, fmt(metrics.get("test_rmse")), fmt(metrics.get("test_nll")), fmt(metrics.get("interval_calibration_mae"))]
        for task_name, config_name, metrics in task_rows
    ]
    hard_headers = ["Setting", "Run", "RMSE", "NLL", "Calib MAE"]
    hard_rows_formatted = [
        [task_name, config_name, fmt(metrics.get("test_rmse")), fmt(metrics.get("test_nll")), fmt(metrics.get("interval_calibration_mae"))]
        for task_name, config_name, metrics in hard_task_rows
    ]
    hard_rows_formatted.append(
        [
            "32x32 multisource",
            hard_multiseed["label"],
            fmt_mean_std(hard_multiseed["rmse_mean"], hard_multiseed["rmse_std"]),
            fmt_mean_std(hard_multiseed["nll_mean"], hard_multiseed["nll_std"]),
            fmt_mean_std(hard_multiseed["calib_mean"], hard_multiseed["calib_std"]),
        ]
    )
    hard_rows_formatted.append(
        [
            "32x32 multisource",
            hard_random40_multiseed["label"],
            fmt_mean_std(hard_random40_multiseed["rmse_mean"], hard_random40_multiseed["rmse_std"]),
            fmt_mean_std(hard_random40_multiseed["nll_mean"], hard_random40_multiseed["nll_std"]),
            fmt_mean_std(hard_random40_multiseed["calib_mean"], hard_random40_multiseed["calib_std"]),
        ]
    )
    structured_headers = ["Method", "RMSE", "NLL", "Calib MAE"]
    structured_rows_formatted = [
        [label, fmt(metrics.get("test_rmse")), fmt(metrics.get("test_nll")), fmt(metrics.get("interval_calibration_mae"))]
        for label, metrics in structured_rows
    ]
    realistic_headers = ["Method", "Sensors", "RMSE", "NLL", "Calib MAE"]
    realistic_rows_formatted = [
        [label, str(sensor_count), fmt(metrics.get("test_rmse")), fmt(metrics.get("test_nll")), fmt(metrics.get("interval_calibration_mae"))]
        for label, sensor_count, metrics in realistic_rows
    ]
    realistic_rows_formatted.append(
        [
            realistic_multiseed["label"],
            "64",
            fmt_mean_std(realistic_multiseed["rmse_mean"], realistic_multiseed["rmse_std"]),
            fmt_mean_std(realistic_multiseed["nll_mean"], realistic_multiseed["nll_std"]),
            fmt_mean_std(realistic_multiseed["calib_mean"], realistic_multiseed["calib_std"]),
        ]
    )
    realistic80_headers = ["Method", "Sensors", "RMSE", "NLL", "Calib MAE"]
    realistic80_rows_formatted = [
        [label, str(sensor_count), fmt(metrics.get("test_rmse")), fmt(metrics.get("test_nll")), fmt(metrics.get("interval_calibration_mae"))]
        for label, sensor_count, metrics in realistic80_rows
    ]
    realistic80_rows_formatted.append(
        [
            realistic80_multiseed["label"],
            "56",
            fmt_mean_std(realistic80_multiseed["rmse_mean"], realistic80_multiseed["rmse_std"]),
            fmt_mean_std(realistic80_multiseed["nll_mean"], realistic80_multiseed["nll_std"]),
            fmt_mean_std(realistic80_multiseed["calib_mean"], realistic80_multiseed["calib_std"]),
        ]
    )
    realistic96_headers = ["Method", "Sensors", "RMSE", "NLL", "Calib MAE"]
    realistic96_rows_formatted = [
        [label, str(sensor_count), fmt(metrics.get("test_rmse")), fmt(metrics.get("test_nll")), fmt(metrics.get("interval_calibration_mae"))]
        for label, sensor_count, metrics in realistic96_rows
    ]
    realistic96_rows_formatted.append(
        [
            realistic96_multiseed["label"],
            "72",
            fmt_mean_std(realistic96_multiseed["rmse_mean"], realistic96_multiseed["rmse_std"]),
            fmt_mean_std(realistic96_multiseed["nll_mean"], realistic96_multiseed["nll_std"]),
            fmt_mean_std(realistic96_multiseed["calib_mean"], realistic96_multiseed["calib_std"]),
        ]
    )
    realistic_scale_headers = ["Benchmark", "Configuration", "RMSE", "NLL", "Calib MAE"]
    realistic_scale_rows_formatted = [
        ["48x48 structured multisource", "40 learned sensors", fmt(load_metrics(root, "task_structured48_big40_v2").get("test_rmse")), fmt(load_metrics(root, "task_structured48_big40_v2").get("test_nll")), fmt(load_metrics(root, "task_structured48_big40_v2").get("interval_calibration_mae"))],
        ["64x64 structured realistic", "64 learned sensors", fmt(load_metrics(root, "task_structured_realistic64_main64s").get("test_rmse")), fmt(load_metrics(root, "task_structured_realistic64_main64s").get("test_nll")), fmt(load_metrics(root, "task_structured_realistic64_main64s").get("interval_calibration_mae"))],
        ["64x64 structured realistic", "64 learned sensors (3 seeds)", fmt_mean_std(realistic_multiseed["rmse_mean"], realistic_multiseed["rmse_std"]), fmt_mean_std(realistic_multiseed["nll_mean"], realistic_multiseed["nll_std"]), fmt_mean_std(realistic_multiseed["calib_mean"], realistic_multiseed["calib_std"])],
        ["80x80 structured realistic", "56 learned sensors", fmt(load_metrics(root, "task_structured_realistic80_main").get("test_rmse")), fmt(load_metrics(root, "task_structured_realistic80_main").get("test_nll")), fmt(load_metrics(root, "task_structured_realistic80_main").get("interval_calibration_mae"))],
        ["80x80 structured realistic", "56 learned sensors (3 seeds)", fmt_mean_std(realistic80_multiseed["rmse_mean"], realistic80_multiseed["rmse_std"]), fmt_mean_std(realistic80_multiseed["nll_mean"], realistic80_multiseed["nll_std"]), fmt_mean_std(realistic80_multiseed["calib_mean"], realistic80_multiseed["calib_std"])],
        ["96x96 structured realistic", "72 learned sensors", fmt(load_metrics(root, "task_structured_realistic96_main").get("test_rmse")), fmt(load_metrics(root, "task_structured_realistic96_main").get("test_nll")), fmt(load_metrics(root, "task_structured_realistic96_main").get("interval_calibration_mae"))],
        ["96x96 structured realistic", "72 learned sensors (3 seeds)", fmt_mean_std(realistic96_multiseed["rmse_mean"], realistic96_multiseed["rmse_std"]), fmt_mean_std(realistic96_multiseed["nll_mean"], realistic96_multiseed["nll_std"]), fmt_mean_std(realistic96_multiseed["calib_mean"], realistic96_multiseed["calib_std"])],
    ]

    md_lines = [
        "# Paper-Style Tables",
        "",
        "## Main Task Comparison",
        "",
        markdown_table(main_headers, main_rows),
        "",
        "## Key Ablations",
        "",
        markdown_table(ablation_headers, ablation_rows),
        "",
        "## Task and Scale Generalization",
        "",
        markdown_table(task_headers, task_rows_formatted),
        "",
        "## Hard-Task Progression",
        "",
        markdown_table(hard_headers, hard_rows_formatted),
        "",
        "## Structured 48x48 Comparison",
        "",
        markdown_table(structured_headers, structured_rows_formatted),
        "",
        "## Realistic 64x64 Comparison",
        "",
        markdown_table(realistic_headers, realistic_rows_formatted),
        "",
        "## Realistic 80x80 Comparison",
        "",
        markdown_table(realistic80_headers, realistic80_rows_formatted),
        "",
        "## Realistic 96x96 Comparison",
        "",
        markdown_table(realistic96_headers, realistic96_rows_formatted),
        "",
        "## Realistic Scale-Up",
        "",
        markdown_table(realistic_scale_headers, realistic_scale_rows_formatted),
        "",
        "## Notes",
        "",
        "- Best single-run 20x20 constant result is the 24-sensor learned Gaussian-random model.",
        "- Best stable 20x20 constant result is the 24-sensor three-seed summary.",
        "- On `32x32 multisource`, the learned-sensor line improves from `0.5859` to `0.5164` after scaling the model and sensor budget.",
        "- New matched-budget controls show stronger layout sensitivity on the hard task: `random40` reaches `0.5041` RMSE in the reference seed, while `fixed40` is weaker at `0.5290`.",
        "- The three-seed hard-task summaries are now `0.5285 +/- 0.0107` for `learned40` and `0.5247 +/- 0.0175` for `random40`, so layout ranking is no longer stable across seeds.",
        "- A new 48x48 structured multisource benchmark is now included to move beyond purely smooth Fourier-style coefficient fields.",
        "- The new `64x64 structured realistic` benchmark adds signed forcing, nonzero boundaries, and noisy observations; the learned 64-sensor model reaches `1.2251 +/- 0.0005` RMSE over three seeds.",
        "- The `80x80 structured realistic` benchmark now includes both baselines and a three-seed summary; the current three-seed result is `1.2647 +/- 0.0012` RMSE.",
        "- The `96x96 structured realistic` benchmark now includes random-grid and fixed-grid Bayesian baselines plus a three-seed summary; the current three-seed result is `1.2962 +/- 0.0006` RMSE.",
        "- On the current `96x96 structured realistic` comparison, the deterministic operator is still best in pure RMSE at `1.2959`, while the Bayesian fixed-grid baseline is the strongest uncertainty-aware single run at `1.2965`.",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_sections = [
        "% Auto-generated by scripts/make_paper_tables.py",
        latex_table(
            caption="Main 20x20 constant-forcing comparison.",
            label="tab:main_comparison",
            headers=main_headers,
            rows=main_rows,
        ),
        "",
        latex_table(
            caption="Key ablations on the 20x20 constant-forcing task.",
            label="tab:ablations",
            headers=ablation_headers,
            rows=ablation_rows,
        ),
        "",
        latex_table(
            caption="Task and scale generalization across constant and multisource Darcy settings.",
            label="tab:task_scale",
            headers=task_headers,
            rows=task_rows_formatted,
        ),
        "",
        latex_table(
            caption="Hard-task progression on multisource and structured heterogeneous Darcy settings.",
            label="tab:hard_progression",
            headers=hard_headers,
            rows=hard_rows_formatted,
        ),
        "",
        latex_table(
            caption="Structured 48x48 benchmark comparison.",
            label="tab:structured48_comparison",
            headers=structured_headers,
            rows=structured_rows_formatted,
        ),
        "",
        latex_table(
            caption="Realistic 64x64 benchmark comparison.",
            label="tab:realistic64_comparison",
            headers=realistic_headers,
            rows=realistic_rows_formatted,
        ),
        "",
        latex_table(
            caption="Realistic 80x80 benchmark comparison.",
            label="tab:realistic80_comparison",
            headers=realistic80_headers,
            rows=realistic80_rows_formatted,
        ),
        "",
        latex_table(
            caption="Realistic 96x96 benchmark comparison.",
            label="tab:realistic96_comparison",
            headers=realistic96_headers,
            rows=realistic96_rows_formatted,
        ),
        "",
        latex_table(
            caption="Realistic benchmark scale-up from 48x48 to 96x96.",
            label="tab:realistic_scaleup",
            headers=realistic_scale_headers,
            rows=realistic_scale_rows_formatted,
        ),
        "",
    ]
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(tex_sections), encoding="utf-8")

    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
