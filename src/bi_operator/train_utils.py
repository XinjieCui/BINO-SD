from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


def gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    variance = logvar.exp().clamp_min(1e-6)
    return 0.5 * (((target - mean) ** 2) / variance + logvar).mean()


def coefficient_rmse(log_mean: torch.Tensor, coefficient_target: torch.Tensor) -> torch.Tensor:
    prediction = log_mean.exp()
    return torch.sqrt(torch.mean((prediction - coefficient_target) ** 2))


def sensor_peak_locations(sensor_maps: torch.Tensor) -> list[tuple[int, int]]:
    flat_indices = sensor_maps.view(sensor_maps.shape[0], -1).argmax(dim=1).tolist()
    width = sensor_maps.shape[-1]
    return [(index // width, index % width) for index in flat_indices]


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
