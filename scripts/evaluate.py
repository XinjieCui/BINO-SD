from __future__ import annotations

import argparse
import os
import sys
from statistics import NormalDist
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bi_operator.data import create_dataloader
from bi_operator.model import build_model
from bi_operator.train_utils import coefficient_rmse, gaussian_nll, save_json, sensor_peak_locations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Bayesian inverse operator.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(raw_device: str) -> torch.device:
    if raw_device != "auto":
        return torch.device(raw_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lognormal_std(log_mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    variance = logvar.exp()
    return torch.sqrt((torch.exp(variance) - 1.0) * torch.exp(2.0 * log_mean + variance))


def empirical_coverage(
    coefficient: torch.Tensor,
    log_mean: torch.Tensor,
    logvar: torch.Tensor,
    level: float,
) -> float:
    normal = NormalDist()
    z_value = normal.inv_cdf(0.5 + level / 2.0)
    std = torch.sqrt(logvar.exp())
    lower = torch.exp(log_mean - z_value * std)
    upper = torch.exp(log_mean + z_value * std)
    covered = ((coefficient >= lower) & (coefficient <= upper)).float().mean()
    return float(covered)


def safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).mean()) * (y.pow(2).mean())).clamp_min(1e-8)
    return float((x * y).mean() / denom)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_name = config.get("model_name", "bayesian_operator")
    is_bayesian = model_name == "bayesian_operator"
    model = build_model(
        model_name=model_name,
        grid_size=checkpoint["grid_size"],
        num_sensors=config["num_sensors"],
        hidden_width=config["hidden_width"],
        modes=config["modes"],
        depth=config["depth"],
        sensor_strategy=config.get("sensor_strategy", "learned_softmax"),
        sensor_temperature=config.get("sensor_temperature", 0.2),
        sensor_repulsion_scale=config.get("sensor_repulsion_scale", 0.02),
        sensor_sigma=config.get("sensor_sigma", 0.075),
        sensor_seed=config.get("sensor_seed", 7),
        use_forcing_sensor_fusion=config.get("use_forcing_sensor_fusion", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_loader = create_dataloader(
        dataset_path=args.dataset,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
    )

    totals = {"nll": 0.0, "rmse": 0.0}
    total_count = 0
    example_batch = None
    all_coefficients = []
    all_log_means = []
    all_logvars = []

    with torch.no_grad():
        for batch in test_loader:
            solution = batch["solution"].to(device)
            coefficient = batch["coefficient"].to(device)
            log_coefficient = batch["log_coefficient"].to(device)
            forcing = batch["forcing"].to(device)
            outputs = model(solution, forcing)

            if is_bayesian:
                nll = gaussian_nll(outputs["posterior_mean"], outputs["posterior_logvar"], log_coefficient)
            else:
                nll = torch.zeros((), device=device, dtype=outputs["posterior_mean"].dtype)
            rmse = coefficient_rmse(outputs["posterior_mean"], coefficient)

            batch_size = solution.shape[0]
            total_count += batch_size
            totals["nll"] += float(nll) * batch_size
            totals["rmse"] += float(rmse) * batch_size
            all_coefficients.append(coefficient.cpu())
            all_log_means.append(outputs["posterior_mean"].cpu())
            all_logvars.append(outputs["posterior_logvar"].cpu())

            if example_batch is None:
                example_batch = {
                    "solution": solution[:4].cpu(),
                    "forcing": forcing[:4].cpu(),
                    "coefficient": coefficient[:4].cpu(),
                    "posterior_mean": outputs["posterior_mean"][:4].cpu(),
                    "posterior_logvar": outputs["posterior_logvar"][:4].cpu(),
                    "sensor_maps": outputs["sensor_maps"].cpu(),
                }

    averaged = {name: value / max(1, total_count) for name, value in totals.items()}
    coefficient_tensor = torch.cat(all_coefficients, dim=0)
    log_mean_tensor = torch.cat(all_log_means, dim=0)
    logvar_tensor = torch.cat(all_logvars, dim=0)
    coefficient_mean_tensor = log_mean_tensor.exp()
    coefficient_std_tensor = lognormal_std(log_mean_tensor, logvar_tensor)
    abs_error_tensor = (coefficient_mean_tensor - coefficient_tensor).abs()
    if is_bayesian:
        coverage_levels = [0.5, 0.8, 0.9, 0.95]
        coverage_points = [
            {
                "nominal": level,
                "empirical": empirical_coverage(
                    coefficient=coefficient_tensor,
                    log_mean=log_mean_tensor,
                    logvar=logvar_tensor,
                    level=level,
                ),
            }
            for level in coverage_levels
        ]
        calibration_error = sum(
            abs(point["empirical"] - point["nominal"]) for point in coverage_points
        ) / len(coverage_points)
        sharpness = float(coefficient_std_tensor.mean())
        uncertainty_error_corr = safe_corrcoef(coefficient_std_tensor, abs_error_tensor)
    else:
        coverage_points = []
        calibration_error = None
        sharpness = None
        uncertainty_error_corr = None
    sensor_locations = sensor_peak_locations(example_batch["sensor_maps"])
    save_json(
        args.output_dir / "evaluation_metrics.json",
        {
            "test_nll": averaged["nll"] if is_bayesian else None,
            "test_rmse": averaged["rmse"],
            "model_name": model_name,
            "sensor_strategy": config.get("sensor_strategy", "learned_softmax"),
            "sharpness_mean_std": sharpness,
            "uncertainty_error_corr": uncertainty_error_corr,
            "interval_calibration_mae": calibration_error,
            "coverage_points": coverage_points,
            "sensor_locations": sensor_locations,
        },
    )

    num_examples = int(example_batch["solution"].shape[0])
    num_rows = max(1, min(4, num_examples))
    figure, axes = plt.subplots(num_rows, 6, figsize=(18, 3 * num_rows), constrained_layout=True)
    if num_rows == 1:
        axes = axes[None, :]
    for row in range(num_rows):
        solution = example_batch["solution"][row, 0].numpy()
        forcing = example_batch["forcing"][row, 0].numpy()
        coefficient = example_batch["coefficient"][row, 0].numpy()
        coefficient_pred = example_batch["posterior_mean"][row, 0].exp().numpy()
        uncertainty = example_batch["posterior_logvar"][row, 0].exp().sqrt().numpy()
        abs_error = abs(coefficient_pred - coefficient)

        axes[row, 0].imshow(solution, origin="lower", cmap="viridis")
        axes[row, 0].set_title("Solution")
        axes[row, 1].imshow(coefficient, origin="lower", cmap="magma")
        axes[row, 1].set_title("True Coefficient")
        axes[row, 2].imshow(coefficient_pred, origin="lower", cmap="magma")
        axes[row, 2].set_title("Predicted Mean")
        axes[row, 3].imshow(forcing, origin="lower", cmap="plasma")
        axes[row, 3].set_title("Forcing")
        axes[row, 4].imshow(uncertainty, origin="lower", cmap="cividis")
        axes[row, 4].set_title("Log Std")
        axes[row, 5].imshow(abs_error, origin="lower", cmap="inferno")
        axes[row, 5].set_title("Abs Error")

        for col in range(6):
            axes[row, col].scatter(
                [location[1] for location in sensor_locations],
                [location[0] for location in sensor_locations],
                c="red",
                s=18,
                marker="x",
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    figure.savefig(args.output_dir / "qualitative_results.png", dpi=180)
    plt.close(figure)

    if is_bayesian:
        calibration_figure, calibration_axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
        calibration_axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Ideal")
        calibration_axis.plot(
            [point["nominal"] for point in coverage_points],
            [point["empirical"] for point in coverage_points],
            marker="o",
            color="tab:blue",
            label="Empirical",
        )
        calibration_axis.set_xlim(0.0, 1.0)
        calibration_axis.set_ylim(0.0, 1.0)
        calibration_axis.set_xlabel("Nominal Coverage")
        calibration_axis.set_ylabel("Empirical Coverage")
        calibration_axis.set_title("Interval Calibration")
        calibration_axis.legend()
        calibration_figure.savefig(args.output_dir / "uncertainty_calibration.png", dpi=180)
        plt.close(calibration_figure)

    if is_bayesian:
        print(f"Test NLL: {averaged['nll']:.4f}")
    print(f"Test RMSE: {averaged['rmse']:.4f}")
    if is_bayesian:
        print(f"Sharpness mean std: {sharpness:.4f}")
        print(f"Uncertainty-error corr: {uncertainty_error_corr:.4f}")
        print(f"Interval calibration MAE: {calibration_error:.4f}")
    print(f"Sensor locations: {sensor_locations}")
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
