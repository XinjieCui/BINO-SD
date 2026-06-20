from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bi_operator.data import create_dataloader
from bi_operator.model import build_model
from bi_operator.pde import darcy_residual_torch
from bi_operator.train_utils import coefficient_rmse, gaussian_nll, save_json, sensor_peak_locations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Bayesian inverse operator.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="bayesian_operator",
        choices=["bayesian_operator", "deterministic_operator", "unet_deterministic"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-sensors", type=int, default=8)
    parser.add_argument("--hidden-width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--sensor-strategy",
        type=str,
        default="learned_softmax",
        choices=[
            "learned_softmax",
            "learned_gaussian",
            "learned_gaussian_random",
            "fixed_grid",
            "random_grid",
        ],
    )
    parser.add_argument("--sensor-temperature", type=float, default=0.2)
    parser.add_argument("--sensor-repulsion-scale", type=float, default=0.02)
    parser.add_argument("--sensor-sigma", type=float, default=0.075)
    parser.add_argument("--sensor-seed", type=int, default=7)
    parser.add_argument("--use-forcing-sensor-fusion", action="store_true")
    parser.add_argument("--coefficient-weight", type=float, default=1.0)
    parser.add_argument("--log-weight", type=float, default=0.25)
    parser.add_argument("--uncertainty-weight", type=float, default=0.05)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=5e-4)
    parser.add_argument("--diversity-weight", type=float, default=0.2)
    parser.add_argument("--repulsion-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device: str) -> torch.device:
    if raw_device != "auto":
        return torch.device(raw_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: BayesianInverseOperator,
    dataloader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    coefficient_weight: float,
    log_weight: float,
    uncertainty_weight: float,
    physics_weight: float,
    entropy_weight: float,
    diversity_weight: float,
    repulsion_weight: float,
    use_bayesian_loss: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "nll": 0.0,
        "coefficient_mse": 0.0,
        "log_mse": 0.0,
        "physics": 0.0,
        "rmse": 0.0,
        "repulsion": 0.0,
    }
    total_count = 0

    for batch in dataloader:
        solution = batch["solution"].to(device)
        coefficient = batch["coefficient"].to(device)
        log_coefficient = batch["log_coefficient"].to(device)
        forcing = batch["forcing"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(solution, forcing)
            posterior_mean = outputs["posterior_mean"]
            posterior_logvar = outputs["posterior_logvar"]
            coefficient_mean = posterior_mean.exp()

            if use_bayesian_loss:
                nll = gaussian_nll(posterior_mean, posterior_logvar, log_coefficient)
            else:
                nll = torch.zeros((), device=device, dtype=posterior_mean.dtype)
            coefficient_mse = F.mse_loss(coefficient_mean, coefficient)
            log_mse = F.mse_loss(posterior_mean, log_coefficient)
            physics = darcy_residual_torch(
                coefficient_mean[:, 0],
                solution[:, 0],
                forcing=forcing[:, 0],
            ).pow(2).mean()
            total_loss = (
                coefficient_weight * coefficient_mse
                + log_weight * log_mse
                + uncertainty_weight * nll
                + physics_weight * physics
                + entropy_weight * outputs["entropy_penalty"]
                + diversity_weight * outputs["diversity_penalty"]
                + repulsion_weight * outputs["repulsion_penalty"]
            )
            rmse = coefficient_rmse(posterior_mean, coefficient)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = solution.shape[0]
        total_count += batch_size
        totals["loss"] += float(total_loss.detach()) * batch_size
        totals["nll"] += float(nll.detach()) * batch_size
        totals["coefficient_mse"] += float(coefficient_mse.detach()) * batch_size
        totals["log_mse"] += float(log_mse.detach()) * batch_size
        totals["physics"] += float(physics.detach()) * batch_size
        totals["rmse"] += float(rmse.detach()) * batch_size
        totals["repulsion"] += float(outputs["repulsion_penalty"].detach()) * batch_size

    return {name: value / max(1, total_count) for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    train_loader = create_dataloader(
        dataset_path=args.dataset,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        dataset_path=args.dataset,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
    )

    sample_batch = next(iter(train_loader))
    grid_size = sample_batch["solution"].shape[-1]
    model = build_model(
        model_name=args.model_name,
        grid_size=grid_size,
        num_sensors=args.num_sensors,
        hidden_width=args.hidden_width,
        modes=args.modes,
        depth=args.depth,
        sensor_strategy=args.sensor_strategy,
        sensor_temperature=args.sensor_temperature,
        sensor_repulsion_scale=args.sensor_repulsion_scale,
        sensor_sigma=args.sensor_sigma,
        sensor_seed=args.sensor_seed,
        use_forcing_sensor_fusion=args.use_forcing_sensor_fusion,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    best_checkpoint_path = args.output_dir / "best_model.pth"
    start_time = time.time()
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    use_bayesian_loss = args.model_name == "bayesian_operator"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            coefficient_weight=args.coefficient_weight,
            log_weight=args.log_weight,
            uncertainty_weight=args.uncertainty_weight,
            physics_weight=args.physics_weight,
            entropy_weight=args.entropy_weight,
            diversity_weight=args.diversity_weight,
            repulsion_weight=args.repulsion_weight,
            use_bayesian_loss=use_bayesian_loss,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            device=device,
            coefficient_weight=args.coefficient_weight,
            log_weight=args.log_weight,
            uncertainty_weight=args.uncertainty_weight,
            physics_weight=args.physics_weight,
            entropy_weight=args.entropy_weight,
            diversity_weight=args.diversity_weight,
            repulsion_weight=args.repulsion_weight,
            use_bayesian_loss=use_bayesian_loss,
        )
        scheduler.step()

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_nll": train_metrics["nll"],
            "train_coefficient_mse": train_metrics["coefficient_mse"],
            "train_log_mse": train_metrics["log_mse"],
            "train_physics": train_metrics["physics"],
            "train_rmse": train_metrics["rmse"],
            "train_repulsion": train_metrics["repulsion"],
            "val_loss": val_metrics["loss"],
            "val_nll": val_metrics["nll"],
            "val_coefficient_mse": val_metrics["coefficient_mse"],
            "val_log_mse": val_metrics["log_mse"],
            "val_physics": val_metrics["physics"],
            "val_rmse": val_metrics["rmse"],
            "val_repulsion": val_metrics["repulsion"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f}"
        )

        is_better = (
            val_metrics["rmse"] < best_val_rmse - 1e-6
            or (
                abs(val_metrics["rmse"] - best_val_rmse) <= 1e-6
                and val_metrics["loss"] < best_val_loss
            )
        )
        if is_better:
            best_val_loss = val_metrics["loss"]
            best_val_rmse = val_metrics["rmse"]
            sensor_maps = model.sensor_layer.weight_maps().detach().cpu()
            checkpoint = {
                "model_state": model.state_dict(),
                "config": serializable_config,
                "grid_size": grid_size,
                "sensor_locations": sensor_peak_locations(sensor_maps),
                "best_val_rmse": best_val_rmse,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, best_checkpoint_path)

    elapsed = time.time() - start_time
    sensor_maps = model.sensor_layer.weight_maps().detach().cpu()
    summary = {
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "sensor_locations": sensor_peak_locations(sensor_maps),
        "history": history,
    }

    save_json(args.output_dir / "training_history.json", {"history": history})
    save_json(args.output_dir / "run_summary.json", summary)

    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Learned sensor locations: {summary['sensor_locations']}")
    print(f"Training time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
