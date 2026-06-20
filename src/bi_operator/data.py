from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader, Dataset


class DarcyInverseDataset(Dataset):
    def __init__(self, dataset_path: str | Path, split: str) -> None:
        data = np.load(Path(dataset_path))
        self.coefficients = data[f"{split}_coefficients"].astype(np.float32)
        self.solutions = data[f"{split}_solutions"].astype(np.float32)
        if f"{split}_clean_solutions" in data:
            self.clean_solutions = data[f"{split}_clean_solutions"].astype(np.float32)
        else:
            self.clean_solutions = self.solutions
        self.log_coefficients = np.log(self.coefficients).astype(np.float32)
        if f"{split}_forcings" in data:
            self.forcings = data[f"{split}_forcings"].astype(np.float32)
        else:
            forcing_value = float(data["forcing"]) if "forcing" in data else 1.0
            self.forcings = np.full_like(self.solutions, forcing_value, dtype=np.float32)
        if f"{split}_boundaries" in data:
            self.boundaries = data[f"{split}_boundaries"].astype(np.float32)
        else:
            self.boundaries = np.zeros_like(self.solutions, dtype=np.float32)
        self.task_name = str(data["task"]) if "task" in data else "darcy_constant"
        self.observation_noise_std = (
            float(data["observation_noise_std"]) if "observation_noise_std" in data else 0.0
        )

    def __len__(self) -> int:
        return int(self.coefficients.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "coefficient": torch.from_numpy(self.coefficients[index]).unsqueeze(0),
            "log_coefficient": torch.from_numpy(self.log_coefficients[index]).unsqueeze(0),
            "solution": torch.from_numpy(self.solutions[index]).unsqueeze(0),
            "clean_solution": torch.from_numpy(self.clean_solutions[index]).unsqueeze(0),
            "forcing": torch.from_numpy(self.forcings[index]).unsqueeze(0),
            "boundary": torch.from_numpy(self.boundaries[index]).unsqueeze(0),
        }


def create_dataloader(
    dataset_path: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = DarcyInverseDataset(dataset_path=dataset_path, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
