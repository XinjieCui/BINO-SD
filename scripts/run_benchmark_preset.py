from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate.py"


def bayesian_realistic_config(
    *,
    dataset: Path,
    output_dir: Path,
    seed: int,
    sensor_strategy: str = "learned_gaussian_random",
    trainable_sensor: bool = True,
    epochs: int = 24,
    batch_size: int = 8,
    hidden_width: int = 104,
    modes: int = 14,
    depth: int = 6,
    num_sensors: int = 48,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    sensor_sigma: float = 0.032,
    sensor_repulsion_scale: float = 0.006,
    coefficient_weight: float = 1.0,
    log_weight: float = 0.25,
    uncertainty_weight: float = 0.04,
    physics_weight: float = 0.05,
    diversity_weight: float = 0.12,
    repulsion_weight: float = 0.08,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "output_dir": output_dir,
        "train": {
            "model_name": "bayesian_operator",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_sensors": num_sensors,
            "hidden_width": hidden_width,
            "modes": modes,
            "depth": depth,
            "sensor_strategy": sensor_strategy,
            "sensor_sigma": sensor_sigma,
            "sensor_repulsion_scale": sensor_repulsion_scale,
            "coefficient_weight": coefficient_weight,
            "log_weight": log_weight,
            "uncertainty_weight": uncertainty_weight,
            "physics_weight": physics_weight,
            "diversity_weight": diversity_weight if trainable_sensor else 0.0,
            "repulsion_weight": repulsion_weight if trainable_sensor else 0.0,
            "seed": seed,
            "sensor_seed": seed,
        },
    }


def deterministic_realistic_config(
    *,
    dataset: Path,
    output_dir: Path,
    seed: int,
    model_name: str = "deterministic_operator",
    sensor_strategy: str = "random_grid",
    epochs: int = 24,
    batch_size: int = 8,
    hidden_width: int = 104,
    modes: int = 14,
    depth: int = 6,
    num_sensors: int = 48,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    sensor_sigma: float = 0.032,
    sensor_repulsion_scale: float = 0.006,
    coefficient_weight: float = 1.0,
    log_weight: float = 0.25,
    physics_weight: float = 0.05,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "output_dir": output_dir,
        "train": {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_sensors": num_sensors,
            "hidden_width": hidden_width,
            "modes": modes,
            "depth": depth,
            "sensor_strategy": sensor_strategy,
            "sensor_sigma": sensor_sigma,
            "sensor_repulsion_scale": sensor_repulsion_scale,
            "coefficient_weight": coefficient_weight,
            "log_weight": log_weight,
            "uncertainty_weight": 0.0,
            "physics_weight": physics_weight,
            "diversity_weight": 0.0,
            "repulsion_weight": 0.0,
            "seed": seed,
            "sensor_seed": seed,
        },
    }


def bayesian_multisource32_config(
    *,
    output_dir: Path,
    seed: int,
    sensor_strategy: str = "learned_gaussian_random",
    trainable_sensor: bool = True,
    epochs: int = 50,
    batch_size: int = 24,
    hidden_width: int = 80,
    modes: int = 12,
    depth: int = 6,
    num_sensors: int = 40,
    learning_rate: float = 4e-4,
    weight_decay: float = 5e-4,
    sensor_sigma: float = 0.045,
    sensor_repulsion_scale: float = 0.008,
    coefficient_weight: float = 1.0,
    log_weight: float = 0.25,
    uncertainty_weight: float = 0.03,
    physics_weight: float = 0.05,
    entropy_weight: float = 1e-3,
    diversity_weight: float = 0.05,
    repulsion_weight: float = 0.1,
) -> dict[str, object]:
    return {
        "dataset": MULTISOURCE32_DATASET,
        "output_dir": output_dir,
        "train": {
            "model_name": "bayesian_operator",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_sensors": num_sensors,
            "hidden_width": hidden_width,
            "modes": modes,
            "depth": depth,
            "sensor_strategy": sensor_strategy,
            "sensor_sigma": sensor_sigma,
            "sensor_repulsion_scale": sensor_repulsion_scale,
            "coefficient_weight": coefficient_weight,
            "log_weight": log_weight,
            "uncertainty_weight": uncertainty_weight,
            "physics_weight": physics_weight,
            "entropy_weight": entropy_weight,
            "diversity_weight": diversity_weight if trainable_sensor else 0.0,
            "repulsion_weight": repulsion_weight if trainable_sensor else 0.0,
            "seed": seed,
            "sensor_seed": seed,
        },
    }


MULTISOURCE32_DATASET = PROJECT_ROOT / "data" / "darcy_multisource_32_large.npz"
REALISTIC64_DATASET = PROJECT_ROOT / "data" / "darcy_structured_realistic_64_medium.npz"
REALISTIC80_DATASET = PROJECT_ROOT / "data" / "darcy_structured_realistic_80_medium.npz"
REALISTIC96_DATASET = PROJECT_ROOT / "data" / "darcy_structured_realistic_96_small.npz"


PRESETS = {
    "multisource32_random40": bayesian_multisource32_config(
        output_dir=PROJECT_ROOT / "artifacts" / "task_multisource32_random40",
        seed=7,
        sensor_strategy="random_grid",
        trainable_sensor=False,
    ),
    "multisource32_random40_seed11": bayesian_multisource32_config(
        output_dir=PROJECT_ROOT / "artifacts" / "task_multisource32_random40_seed11",
        seed=11,
        sensor_strategy="random_grid",
        trainable_sensor=False,
    ),
    "multisource32_random40_seed17": bayesian_multisource32_config(
        output_dir=PROJECT_ROOT / "artifacts" / "task_multisource32_random40_seed17",
        seed=17,
        sensor_strategy="random_grid",
        trainable_sensor=False,
    ),
    "multisource32_fixed40": bayesian_multisource32_config(
        output_dir=PROJECT_ROOT / "artifacts" / "task_multisource32_fixed40",
        seed=7,
        sensor_strategy="fixed_grid",
        trainable_sensor=False,
    ),
    "realistic64_sanity": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_sanity",
        seed=13,
        epochs=4,
        batch_size=6,
        hidden_width=96,
        modes=12,
        coefficient_weight=1.0,
        log_weight=0.25,
        uncertainty_weight=0.03,
        physics_weight=0.04,
    ),
    "realistic64_main": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_main",
        seed=13,
    ),
    "realistic64_seed11": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_seed11",
        seed=11,
    ),
    "realistic64_seed17": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_seed17",
        seed=17,
    ),
    "realistic64_main64s": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_main64s",
        seed=13,
        epochs=28,
        batch_size=8,
        hidden_width=112,
        modes=14,
        depth=6,
        num_sensors=64,
        sensor_sigma=0.030,
        sensor_repulsion_scale=0.005,
    ),
    "realistic64_64s_seed11": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_64s_seed11",
        seed=11,
        epochs=28,
        batch_size=8,
        hidden_width=112,
        modes=14,
        depth=6,
        num_sensors=64,
        sensor_sigma=0.030,
        sensor_repulsion_scale=0.005,
    ),
    "realistic64_64s_seed17": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_64s_seed17",
        seed=17,
        epochs=28,
        batch_size=8,
        hidden_width=112,
        modes=14,
        depth=6,
        num_sensors=64,
        sensor_sigma=0.030,
        sensor_repulsion_scale=0.005,
    ),
    "realistic64_random48": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_random48",
        seed=13,
        sensor_strategy="random_grid",
        trainable_sensor=False,
    ),
    "realistic64_fixed48": bayesian_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_fixed48",
        seed=13,
        sensor_strategy="fixed_grid",
        trainable_sensor=False,
    ),
    "realistic64_det48": deterministic_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_det48",
        seed=13,
        model_name="deterministic_operator",
        sensor_strategy="random_grid",
    ),
    "realistic64_unet48": deterministic_realistic_config(
        dataset=REALISTIC64_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic64_unet48",
        seed=13,
        model_name="unet_deterministic",
        sensor_strategy="random_grid",
        hidden_width=96,
        modes=12,
        depth=5,
    ),
    "realistic80_main": bayesian_realistic_config(
        dataset=REALISTIC80_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic80_main",
        seed=17,
        epochs=16,
        batch_size=4,
        hidden_width=112,
        modes=16,
        depth=6,
        num_sensors=56,
        sensor_sigma=0.028,
        sensor_repulsion_scale=0.005,
    ),
    "realistic80_seed11": bayesian_realistic_config(
        dataset=REALISTIC80_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic80_seed11",
        seed=11,
        epochs=16,
        batch_size=4,
        hidden_width=112,
        modes=16,
        depth=6,
        num_sensors=56,
        sensor_sigma=0.028,
        sensor_repulsion_scale=0.005,
    ),
    "realistic80_seed23": bayesian_realistic_config(
        dataset=REALISTIC80_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic80_seed23",
        seed=23,
        epochs=16,
        batch_size=4,
        hidden_width=112,
        modes=16,
        depth=6,
        num_sensors=56,
        sensor_sigma=0.028,
        sensor_repulsion_scale=0.005,
    ),
    "realistic80_random56": bayesian_realistic_config(
        dataset=REALISTIC80_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic80_random56",
        seed=17,
        sensor_strategy="random_grid",
        trainable_sensor=False,
        epochs=16,
        batch_size=4,
        hidden_width=112,
        modes=16,
        depth=6,
        num_sensors=56,
        sensor_sigma=0.028,
        sensor_repulsion_scale=0.005,
    ),
    "realistic80_fixed56": bayesian_realistic_config(
        dataset=REALISTIC80_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic80_fixed56",
        seed=17,
        sensor_strategy="fixed_grid",
        trainable_sensor=False,
        epochs=16,
        batch_size=4,
        hidden_width=112,
        modes=16,
        depth=6,
        num_sensors=56,
        sensor_sigma=0.028,
        sensor_repulsion_scale=0.005,
    ),
    "realistic80_det56": deterministic_realistic_config(
        dataset=REALISTIC80_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic80_det56",
        seed=17,
        model_name="deterministic_operator",
        sensor_strategy="random_grid",
        epochs=16,
        batch_size=4,
        hidden_width=112,
        modes=16,
        depth=6,
        num_sensors=56,
        sensor_sigma=0.028,
        sensor_repulsion_scale=0.005,
    ),
    "realistic80_unet56": deterministic_realistic_config(
        dataset=REALISTIC80_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic80_unet56",
        seed=17,
        model_name="unet_deterministic",
        sensor_strategy="random_grid",
        epochs=16,
        batch_size=4,
        hidden_width=104,
        modes=14,
        depth=5,
        num_sensors=56,
        sensor_sigma=0.028,
        sensor_repulsion_scale=0.005,
    ),
    "realistic96_main": bayesian_realistic_config(
        dataset=REALISTIC96_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic96_main",
        seed=23,
        epochs=12,
        batch_size=2,
        hidden_width=120,
        modes=18,
        depth=6,
        num_sensors=72,
        sensor_sigma=0.024,
        sensor_repulsion_scale=0.0045,
    ),
    "realistic96_seed11": bayesian_realistic_config(
        dataset=REALISTIC96_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic96_seed11",
        seed=11,
        epochs=12,
        batch_size=2,
        hidden_width=120,
        modes=18,
        depth=6,
        num_sensors=72,
        sensor_sigma=0.024,
        sensor_repulsion_scale=0.0045,
    ),
    "realistic96_seed37": bayesian_realistic_config(
        dataset=REALISTIC96_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic96_seed37",
        seed=37,
        epochs=12,
        batch_size=2,
        hidden_width=120,
        modes=18,
        depth=6,
        num_sensors=72,
        sensor_sigma=0.024,
        sensor_repulsion_scale=0.0045,
    ),
    "realistic96_random72": bayesian_realistic_config(
        dataset=REALISTIC96_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic96_random72",
        seed=23,
        sensor_strategy="random_grid",
        trainable_sensor=False,
        epochs=12,
        batch_size=2,
        hidden_width=120,
        modes=18,
        depth=6,
        num_sensors=72,
        sensor_sigma=0.024,
        sensor_repulsion_scale=0.0045,
    ),
    "realistic96_fixed72": bayesian_realistic_config(
        dataset=REALISTIC96_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic96_fixed72",
        seed=23,
        sensor_strategy="fixed_grid",
        trainable_sensor=False,
        epochs=12,
        batch_size=2,
        hidden_width=120,
        modes=18,
        depth=6,
        num_sensors=72,
        sensor_sigma=0.024,
        sensor_repulsion_scale=0.0045,
    ),
    "realistic96_det72": deterministic_realistic_config(
        dataset=REALISTIC96_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic96_det72",
        seed=23,
        model_name="deterministic_operator",
        sensor_strategy="random_grid",
        epochs=12,
        batch_size=2,
        hidden_width=120,
        modes=18,
        depth=6,
        num_sensors=72,
        sensor_sigma=0.024,
        sensor_repulsion_scale=0.0045,
    ),
    "realistic96_unet72": deterministic_realistic_config(
        dataset=REALISTIC96_DATASET,
        output_dir=PROJECT_ROOT / "artifacts" / "task_structured_realistic96_unet72",
        seed=23,
        model_name="unet_deterministic",
        sensor_strategy="random_grid",
        epochs=12,
        batch_size=2,
        hidden_width=112,
        modes=16,
        depth=5,
        num_sensors=72,
        sensor_sigma=0.024,
        sensor_repulsion_scale=0.0045,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a named benchmark training preset.")
    parser.add_argument("--preset", type=str, required=True, choices=sorted(PRESETS))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-only", action="store_true")
    return parser.parse_args()


def add_train_args(command: list[str], train_config: dict[str, object]) -> list[str]:
    for key, value in train_config.items():
        flag = f"--{key.replace('_', '-')}"
        command.extend([flag, str(value)])
    return command


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    dataset = Path(preset["dataset"])
    output_dir = Path(preset["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = dict(preset["train"])
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.epochs is not None:
        train_config["epochs"] = args.epochs

    train_command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
    ]
    train_command = add_train_args(train_command, train_config)
    subprocess.run(train_command, check=True)

    if args.train_only:
        return

    evaluate_command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--dataset",
        str(dataset),
        "--checkpoint",
        str(output_dir / "best_model.pth"),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(train_config["batch_size"]),
        "--device",
        args.device,
    ]
    subprocess.run(evaluate_command, check=True)


if __name__ == "__main__":
    main()
