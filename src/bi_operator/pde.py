from __future__ import annotations

import math
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def sample_log_coefficient_field(
    grid_size: int,
    rng: np.random.Generator,
    max_modes: int = 4,
    amplitude: float = 0.55,
) -> np.ndarray:
    x = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    field = np.zeros((grid_size, grid_size), dtype=np.float64)

    for mx in range(1, max_modes + 1):
        for my in range(1, max_modes + 1):
            decay = 1.0 / (mx * mx + my * my)
            phase_x = rng.uniform(0.0, 2.0 * math.pi)
            phase_y = rng.uniform(0.0, 2.0 * math.pi)
            weight_s = rng.normal(scale=decay)
            weight_c = rng.normal(scale=0.5 * decay)
            field += weight_s * np.sin(math.pi * mx * xx + phase_x) * np.sin(
                math.pi * my * yy + phase_y
            )
            field += weight_c * np.cos(math.pi * mx * xx + phase_x) * np.cos(
                math.pi * my * yy + phase_y
            )

    field -= field.mean()
    field /= field.std() + 1e-6
    return (amplitude * field).astype(np.float32)


def smooth_field(field: np.ndarray, passes: int = 2) -> np.ndarray:
    smoothed = field.astype(np.float64, copy=True)
    for _ in range(passes):
        smoothed = (
            0.40 * smoothed
            + 0.15 * np.roll(smoothed, 1, axis=0)
            + 0.15 * np.roll(smoothed, -1, axis=0)
            + 0.15 * np.roll(smoothed, 1, axis=1)
            + 0.15 * np.roll(smoothed, -1, axis=1)
        )
    return smoothed


def sample_structured_log_coefficient_field(
    grid_size: int,
    rng: np.random.Generator,
    amplitude: float = 0.55,
) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")

    # Low-frequency background keeps the field correlated at basin scale.
    field = sample_log_coefficient_field(
        grid_size=grid_size,
        rng=rng,
        max_modes=2,
        amplitude=0.20,
    ).astype(np.float64)

    # Add several rotated elliptical inclusions to mimic facies blocks.
    num_inclusions = int(rng.integers(3, 7))
    for _ in range(num_inclusions):
        center_x = rng.uniform(0.12, 0.88)
        center_y = rng.uniform(0.12, 0.88)
        axis_a = rng.uniform(0.06, 0.22)
        axis_b = rng.uniform(0.04, 0.18)
        angle = rng.uniform(0.0, math.pi)
        contrast = rng.uniform(-1.2, 1.4)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dx = xx - center_x
        dy = yy - center_y
        rot_x = cos_a * dx + sin_a * dy
        rot_y = -sin_a * dx + cos_a * dy
        ellipse = (rot_x / axis_a) ** 2 + (rot_y / axis_b) ** 2
        field += contrast * np.exp(-1.8 * ellipse)

    # Add a meandering high/low permeability channel.
    channel_center = rng.uniform(0.28, 0.72)
    channel_amp = rng.uniform(0.08, 0.18)
    channel_freq = rng.integers(1, 4)
    channel_phase = rng.uniform(0.0, 2.0 * math.pi)
    channel_width = rng.uniform(0.04, 0.08)
    channel_contrast = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
    centerline = channel_center + channel_amp * np.sin(2.0 * math.pi * channel_freq * xx + channel_phase)
    distance = np.abs(yy - centerline)
    channel_mask = np.exp(-(distance / channel_width) ** 2)
    field += channel_contrast * channel_mask

    # Blend with a softly thresholded facies map for sharper interfaces.
    facies_seed = smooth_field(rng.normal(size=(grid_size, grid_size)), passes=3)
    facies_seed = np.tanh(1.5 * facies_seed)
    field += 0.35 * facies_seed

    field = smooth_field(field, passes=2)
    field -= field.mean()
    field /= field.std() + 1e-6
    return np.clip(amplitude * field, -1.4, 1.4).astype(np.float32)


def sample_coefficient_field(
    grid_size: int,
    rng: np.random.Generator,
    max_modes: int = 4,
    amplitude: float = 0.55,
) -> np.ndarray:
    log_field = sample_log_coefficient_field(
        grid_size=grid_size,
        rng=rng,
        max_modes=max_modes,
        amplitude=amplitude,
    )
    return np.exp(log_field).astype(np.float32)


def sample_structured_coefficient_field(
    grid_size: int,
    rng: np.random.Generator,
    amplitude: float = 0.55,
) -> np.ndarray:
    log_field = sample_structured_log_coefficient_field(
        grid_size=grid_size,
        rng=rng,
        amplitude=amplitude,
    )
    return np.exp(log_field).astype(np.float32)


def sample_boundary_field(
    grid_size: int,
    rng: np.random.Generator,
    task: str,
) -> np.ndarray:
    boundary = np.zeros((grid_size, grid_size), dtype=np.float64)
    if task != "darcy_structured_realistic":
        return boundary.astype(np.float32)

    coords = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)

    def profile(scale: float = 0.18) -> np.ndarray:
        base = rng.uniform(-0.6, 0.6) * scale
        slope = rng.uniform(-1.0, 1.0) * scale
        amp_1 = rng.uniform(0.3, 1.0) * scale
        amp_2 = rng.uniform(0.1, 0.6) * scale
        freq_1 = int(rng.integers(1, 4))
        freq_2 = int(rng.integers(1, 3))
        phase_1 = rng.uniform(0.0, 2.0 * math.pi)
        phase_2 = rng.uniform(0.0, 2.0 * math.pi)
        return (
            base
            + slope * (coords - 0.5)
            + amp_1 * np.sin(freq_1 * math.pi * coords + phase_1)
            + amp_2 * np.cos(freq_2 * math.pi * coords + phase_2)
        )

    west = profile()
    east = profile()
    south = profile()
    north = profile()

    boundary[0, :] = west
    boundary[-1, :] = east
    boundary[:, 0] = south
    boundary[:, -1] = north

    boundary[0, 0] = 0.5 * (west[0] + south[0])
    boundary[0, -1] = 0.5 * (west[-1] + north[0])
    boundary[-1, 0] = 0.5 * (east[0] + south[-1])
    boundary[-1, -1] = 0.5 * (east[-1] + north[-1])
    return boundary.astype(np.float32)


def sample_forcing_field(
    grid_size: int,
    rng: np.random.Generator,
    task: str,
    forcing_scale: float = 1.0,
) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")

    if task == "darcy_constant":
        forcing = np.full((grid_size, grid_size), forcing_scale, dtype=np.float64)
    elif task in {"darcy_multisource", "darcy_structured_multisource"}:
        forcing = np.full((grid_size, grid_size), 0.15 * forcing_scale, dtype=np.float64)
        num_sources = int(rng.integers(2, 5))
        for _ in range(num_sources):
            center_x = rng.uniform(0.12, 0.88)
            center_y = rng.uniform(0.12, 0.88)
            sigma = rng.uniform(0.06, 0.18)
            amplitude = rng.uniform(0.6, 1.6) * forcing_scale
            forcing += amplitude * np.exp(
                -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma)
            )
        forcing /= forcing.mean() + 1e-6
        forcing *= forcing_scale
    elif task == "darcy_structured_realistic":
        forcing = np.zeros((grid_size, grid_size), dtype=np.float64)
        num_sources = int(rng.integers(3, 7))
        for _ in range(num_sources):
            center_x = rng.uniform(0.10, 0.90)
            center_y = rng.uniform(0.10, 0.90)
            sigma_x = rng.uniform(0.04, 0.14)
            sigma_y = rng.uniform(0.04, 0.14)
            amplitude = rng.uniform(0.7, 1.9) * forcing_scale * rng.choice([-1.0, 1.0])
            forcing += amplitude * np.exp(
                -(
                    ((xx - center_x) / sigma_x) ** 2
                    + ((yy - center_y) / sigma_y) ** 2
                )
                / 2.0
            )
        low_freq_phase = rng.uniform(0.0, 2.0 * math.pi)
        low_freq = (
            0.35 * np.sin(2.0 * math.pi * xx + low_freq_phase)
            + 0.25 * np.cos(2.0 * math.pi * yy - low_freq_phase)
            + 0.20 * np.sin(2.0 * math.pi * (xx + yy) + 0.5 * low_freq_phase)
        )
        forcing += forcing_scale * low_freq
        forcing -= forcing.mean()
        forcing /= np.sqrt(np.mean(forcing**2)) + 1e-6
        forcing *= forcing_scale
    else:
        raise ValueError(f"Unsupported task: {task}")

    return forcing.astype(np.float32)


def solve_darcy_dirichlet(
    coefficient: np.ndarray,
    forcing: float | np.ndarray = 1.0,
    boundary: np.ndarray | None = None,
) -> np.ndarray:
    grid_size = coefficient.shape[0]
    interior = grid_size - 2
    if interior < 1:
        raise ValueError("grid_size must be at least 3")

    h = 1.0 / (grid_size - 1)
    system_size = interior * interior
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    if boundary is None:
        boundary_field = np.zeros_like(coefficient, dtype=np.float64)
    else:
        boundary_field = np.asarray(boundary, dtype=np.float64)
        if boundary_field.shape != coefficient.shape:
            raise ValueError("boundary must have the same shape as coefficient")
    if np.isscalar(forcing):
        rhs = np.full(system_size, float(forcing), dtype=np.float64)
    else:
        rhs = np.asarray(forcing, dtype=np.float64)[1:-1, 1:-1].reshape(system_size)

    def idx(i: int, j: int) -> int:
        return (i - 1) * interior + (j - 1)

    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            row = idx(i, j)
            k_center = float(coefficient[i, j])
            k_e = 0.5 * (k_center + float(coefficient[i + 1, j]))
            k_w = 0.5 * (k_center + float(coefficient[i - 1, j]))
            k_n = 0.5 * (k_center + float(coefficient[i, j + 1]))
            k_s = 0.5 * (k_center + float(coefficient[i, j - 1]))

            diag = (k_e + k_w + k_n + k_s) / (h * h)
            rows.append(row)
            cols.append(row)
            values.append(diag)

            if i + 1 <= grid_size - 2:
                rows.append(row)
                cols.append(idx(i + 1, j))
                values.append(-k_e / (h * h))
            else:
                rhs[row] += (k_e / (h * h)) * boundary_field[i + 1, j]
            if i - 1 >= 1:
                rows.append(row)
                cols.append(idx(i - 1, j))
                values.append(-k_w / (h * h))
            else:
                rhs[row] += (k_w / (h * h)) * boundary_field[i - 1, j]
            if j + 1 <= grid_size - 2:
                rows.append(row)
                cols.append(idx(i, j + 1))
                values.append(-k_n / (h * h))
            else:
                rhs[row] += (k_n / (h * h)) * boundary_field[i, j + 1]
            if j - 1 >= 1:
                rows.append(row)
                cols.append(idx(i, j - 1))
                values.append(-k_s / (h * h))
            else:
                rhs[row] += (k_s / (h * h)) * boundary_field[i, j - 1]

    matrix = csr_matrix((values, (rows, cols)), shape=(system_size, system_size))
    interior_solution = spsolve(matrix, rhs).reshape(interior, interior)
    solution = boundary_field.astype(np.float32, copy=True)
    solution[1:-1, 1:-1] = interior_solution.astype(np.float32)
    return solution


def apply_interior_observation_noise(
    solution: np.ndarray,
    boundary: np.ndarray,
    rng: np.random.Generator,
    relative_std: float,
) -> np.ndarray:
    noisy = solution.astype(np.float32, copy=True)
    if relative_std <= 0.0:
        return noisy

    interior = noisy[1:-1, 1:-1]
    scale = float(interior.std()) * relative_std
    if scale <= 0.0:
        return noisy

    interior += rng.normal(loc=0.0, scale=scale, size=interior.shape).astype(np.float32)
    noisy[0, :] = boundary[0, :]
    noisy[-1, :] = boundary[-1, :]
    noisy[:, 0] = boundary[:, 0]
    noisy[:, -1] = boundary[:, -1]
    return noisy


def generate_darcy_sample(
    grid_size: int,
    rng: np.random.Generator,
    task: str = "darcy_constant",
    forcing_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if task in {"darcy_structured_multisource", "darcy_structured_realistic"}:
        coefficient = sample_structured_coefficient_field(grid_size=grid_size, rng=rng)
    else:
        coefficient = sample_coefficient_field(grid_size=grid_size, rng=rng)
    forcing = sample_forcing_field(
        grid_size=grid_size,
        rng=rng,
        task=task,
        forcing_scale=forcing_scale,
    )
    boundary = sample_boundary_field(grid_size=grid_size, rng=rng, task=task)
    solution = solve_darcy_dirichlet(coefficient=coefficient, forcing=forcing, boundary=boundary)
    return (
        coefficient.astype(np.float32),
        solution.astype(np.float32),
        forcing.astype(np.float32),
        boundary.astype(np.float32),
    )


def darcy_residual_torch(
    coefficient: torch.Tensor,
    solution: torch.Tensor,
    forcing: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    import torch

    if coefficient.ndim != 3 or solution.ndim != 3:
        raise ValueError("coefficient and solution must have shape [batch, height, width]")

    h = 1.0 / (coefficient.shape[-1] - 1)
    k_center = coefficient[:, 1:-1, 1:-1]
    u_center = solution[:, 1:-1, 1:-1]

    k_e = 0.5 * (k_center + coefficient[:, 2:, 1:-1])
    k_w = 0.5 * (k_center + coefficient[:, :-2, 1:-1])
    k_n = 0.5 * (k_center + coefficient[:, 1:-1, 2:])
    k_s = 0.5 * (k_center + coefficient[:, 1:-1, :-2])

    div_term = (
        k_e * (solution[:, 2:, 1:-1] - u_center)
        - k_w * (u_center - solution[:, :-2, 1:-1])
        + k_n * (solution[:, 1:-1, 2:] - u_center)
        - k_s * (u_center - solution[:, 1:-1, :-2])
    ) / (h * h)

    if torch.is_tensor(forcing):
        forcing_term = forcing[:, 1:-1, 1:-1]
    else:
        forcing_term = forcing
    return -div_term - forcing_term
