from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_coordinate_grid(
    batch_size: int,
    grid_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((xx, yy), dim=0)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)


def make_initial_centers(
    num_sensors: int,
    strategy: str,
    seed: int,
) -> torch.Tensor:
    if strategy == "fixed_grid":
        side = math.ceil(math.sqrt(num_sensors))
        coords = torch.linspace(0.08, 0.92, side)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        centers = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)[:num_sensors]
    elif strategy == "random_grid":
        generator = torch.Generator().manual_seed(seed)
        centers = torch.rand(num_sensors, 2, generator=generator) * 0.84 + 0.08
    else:
        raise ValueError(f"Unsupported initialization strategy: {strategy}")
    return centers


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        scale = 1.0 / max(1, in_channels * out_channels)
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

    def compl_mul2d(self, input_ft: torch.Tensor) -> torch.Tensor:
        weights = torch.complex(self.weight_real, self.weight_imag)
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2]
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class BaseSensorLayer(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_sensors: int,
        repulsion_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_sensors = num_sensors
        self.repulsion_scale = repulsion_scale

    def weight_maps(self) -> torch.Tensor:
        raise NotImplementedError

    def centers(self, weights: torch.Tensor) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, self.grid_size, device=weights.device, dtype=weights.dtype)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        center_x = (weights * xx.unsqueeze(0)).sum(dim=(-1, -2))
        center_y = (weights * yy.unsqueeze(0)).sum(dim=(-1, -2))
        return torch.stack((center_x, center_y), dim=-1)

    def penalties(self, weights: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = weights.view(self.num_sensors, -1)
        gram = flat @ flat.t()
        off_diag = gram - torch.diag_embed(torch.diag(gram))
        diversity_penalty = off_diag.pow(2).mean()

        pairwise_diff = centers[:, None, :] - centers[None, :, :]
        pairwise_dist_sq = pairwise_diff.pow(2).sum(dim=-1)
        repulsion_kernel = torch.exp(-pairwise_dist_sq / max(self.repulsion_scale, 1e-6))
        repulsion_penalty = (
            repulsion_kernel - torch.diag_embed(torch.diag(repulsion_kernel))
        ).mean()

        entropy_penalty = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(
            dim=(-1, -2)
        ).mean()
        return diversity_penalty, repulsion_penalty, entropy_penalty

    def encode(
        self,
        solution: torch.Tensor,
        forcing: torch.Tensor | None = None,
        use_forcing_sensor_fusion: bool = False,
    ) -> dict[str, torch.Tensor]:
        field = solution[:, 0]
        weights = self.weight_maps()
        observations = torch.einsum("bhw,khw->bk", field, weights)
        sensor_channels = observations[:, :, None, None] * weights[None, :, :, :]
        sensor_maps = weights.unsqueeze(0).expand(field.shape[0], -1, -1, -1)
        observation_grids = observations[:, :, None, None].expand(
            -1,
            -1,
            self.grid_size,
            self.grid_size,
        )
        coverage = weights.sum(dim=0, keepdim=True).expand(field.shape[0], 1, -1, -1)
        centers = self.centers(weights)
        diversity_penalty, repulsion_penalty, entropy_penalty = self.penalties(weights, centers)
        feature_parts = [sensor_channels, sensor_maps, observation_grids]

        if use_forcing_sensor_fusion:
            if forcing is None:
                forcing_field = torch.ones_like(field)
            else:
                forcing_field = forcing[:, 0]
            forcing_observations = torch.einsum("bhw,khw->bk", forcing_field, weights)
            forcing_channels = forcing_observations[:, :, None, None] * weights[None, :, :, :]
            forcing_grids = forcing_observations[:, :, None, None].expand(
                -1,
                -1,
                self.grid_size,
                self.grid_size,
            )
            interaction = (observations * forcing_observations)[:, :, None, None] * weights[None, :, :, :]
            feature_parts.extend((forcing_channels, forcing_grids, interaction))
        else:
            forcing_observations = None

        features = torch.cat((*feature_parts, coverage), dim=1)
        return {
            "features": features,
            "observations": observations,
            "forcing_observations": forcing_observations,
            "sensor_maps": weights,
            "sensor_centers": centers,
            "diversity_penalty": diversity_penalty,
            "repulsion_penalty": repulsion_penalty,
            "entropy_penalty": entropy_penalty,
        }

    def forward(
        self,
        solution: torch.Tensor,
        forcing: torch.Tensor | None = None,
        use_forcing_sensor_fusion: bool = False,
    ) -> dict[str, torch.Tensor]:
        if solution.ndim != 4 or solution.shape[1] != 1:
            raise ValueError("solution must have shape [batch, 1, height, width]")
        return self.encode(
            solution=solution,
            forcing=forcing,
            use_forcing_sensor_fusion=use_forcing_sensor_fusion,
        )


class SoftSensorLayer(BaseSensorLayer):
    def __init__(
        self,
        grid_size: int,
        num_sensors: int,
        temperature: float = 0.35,
        repulsion_scale: float = 0.02,
    ) -> None:
        super().__init__(grid_size=grid_size, num_sensors=num_sensors, repulsion_scale=repulsion_scale)
        self.temperature = temperature
        self.sensor_logits = nn.Parameter(0.05 * torch.randn(num_sensors, grid_size, grid_size))

    def weight_maps(self) -> torch.Tensor:
        flat = self.sensor_logits.view(self.num_sensors, -1)
        weights = F.softmax(flat / self.temperature, dim=-1)
        return weights.view(self.num_sensors, self.grid_size, self.grid_size)


class GaussianSensorLayer(BaseSensorLayer):
    def __init__(
        self,
        grid_size: int,
        num_sensors: int,
        trainable: bool = True,
        init_strategy: str = "fixed_grid",
        sigma: float = 0.075,
        repulsion_scale: float = 0.02,
        seed: int = 7,
    ) -> None:
        super().__init__(grid_size=grid_size, num_sensors=num_sensors, repulsion_scale=repulsion_scale)
        initial_centers = make_initial_centers(num_sensors=num_sensors, strategy=init_strategy, seed=seed)
        self.sigma = sigma
        self.trainable = trainable
        if trainable:
            logits = torch.logit(initial_centers.clamp(1e-4, 1.0 - 1e-4))
            self.center_logits = nn.Parameter(logits)
        else:
            self.register_buffer("fixed_centers", initial_centers)

    def raw_centers(self) -> torch.Tensor:
        if self.trainable:
            return torch.sigmoid(self.center_logits)
        return self.fixed_centers

    def weight_maps(self) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, self.grid_size, device=self.raw_centers().device)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)
        centers = self.raw_centers()
        dist_sq = (xx - centers[:, 0:1, None]) ** 2 + (yy - centers[:, 1:2, None]) ** 2
        weights = torch.exp(-dist_sq / (2.0 * self.sigma * self.sigma))
        weights = weights / weights.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        return weights

    def centers(self, weights: torch.Tensor) -> torch.Tensor:
        return self.raw_centers()

    def penalties(
        self,
        weights: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diversity_penalty, repulsion_penalty, entropy_penalty = super().penalties(weights, centers)
        if not self.trainable:
            zero = weights.new_zeros(())
            return zero, zero, zero
        return diversity_penalty, repulsion_penalty, entropy_penalty


def build_sensor_layer(
    strategy: str,
    grid_size: int,
    num_sensors: int,
    sensor_temperature: float,
    sensor_repulsion_scale: float,
    sensor_sigma: float,
    sensor_seed: int,
) -> BaseSensorLayer:
    if strategy == "learned_softmax":
        return SoftSensorLayer(
            grid_size=grid_size,
            num_sensors=num_sensors,
            temperature=sensor_temperature,
            repulsion_scale=sensor_repulsion_scale,
        )
    if strategy == "learned_gaussian":
        return GaussianSensorLayer(
            grid_size=grid_size,
            num_sensors=num_sensors,
            trainable=True,
            init_strategy="fixed_grid",
            sigma=sensor_sigma,
            repulsion_scale=sensor_repulsion_scale,
            seed=sensor_seed,
        )
    if strategy == "learned_gaussian_random":
        return GaussianSensorLayer(
            grid_size=grid_size,
            num_sensors=num_sensors,
            trainable=True,
            init_strategy="random_grid",
            sigma=sensor_sigma,
            repulsion_scale=sensor_repulsion_scale,
            seed=sensor_seed,
        )
    if strategy == "fixed_grid":
        return GaussianSensorLayer(
            grid_size=grid_size,
            num_sensors=num_sensors,
            trainable=False,
            init_strategy="fixed_grid",
            sigma=sensor_sigma,
            repulsion_scale=sensor_repulsion_scale,
            seed=sensor_seed,
        )
    if strategy == "random_grid":
        return GaussianSensorLayer(
            grid_size=grid_size,
            num_sensors=num_sensors,
            trainable=False,
            init_strategy="random_grid",
            sigma=sensor_sigma,
            repulsion_scale=sensor_repulsion_scale,
            seed=sensor_seed,
        )
    raise ValueError(f"Unsupported sensor strategy: {strategy}")


class OperatorBackbone(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_sensors: int,
        hidden_width: int = 48,
        modes: int = 8,
        depth: int = 4,
        sensor_strategy: str = "learned_softmax",
        sensor_temperature: float = 0.2,
        sensor_repulsion_scale: float = 0.02,
        sensor_sigma: float = 0.075,
        sensor_seed: int = 7,
        use_forcing_sensor_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.use_forcing_sensor_fusion = use_forcing_sensor_fusion
        self.sensor_layer = build_sensor_layer(
            strategy=sensor_strategy,
            grid_size=grid_size,
            num_sensors=num_sensors,
            sensor_temperature=sensor_temperature,
            sensor_repulsion_scale=sensor_repulsion_scale,
            sensor_sigma=sensor_sigma,
            sensor_seed=sensor_seed,
        )
        feature_multiplier = 6 if use_forcing_sensor_fusion else 3
        self.input_projection = nn.Conv2d(feature_multiplier * num_sensors + 4, hidden_width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(hidden_width, hidden_width, modes, modes) for _ in range(depth)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv2d(hidden_width, hidden_width, kernel_size=1) for _ in range(depth)]
        )
        self.norm_layers = nn.ModuleList([nn.GroupNorm(4, hidden_width) for _ in range(depth)])

    def encode(
        self,
        solution: torch.Tensor,
        forcing: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sensor_outputs = self.sensor_layer(
            solution,
            forcing=forcing,
            use_forcing_sensor_fusion=self.use_forcing_sensor_fusion,
        )
        features = sensor_outputs["features"]
        if forcing is None:
            forcing = torch.ones(
                features.shape[0],
                1,
                self.grid_size,
                self.grid_size,
                device=features.device,
                dtype=features.dtype,
            )
        coords = build_coordinate_grid(
            batch_size=features.shape[0],
            grid_size=self.grid_size,
            device=features.device,
            dtype=features.dtype,
        )
        x = torch.cat((features, forcing, coords), dim=1)
        x = self.input_projection(x)
        for spectral_layer, pointwise_layer, norm_layer in zip(
            self.spectral_layers,
            self.pointwise_layers,
            self.norm_layers,
        ):
            x = spectral_layer(x) + pointwise_layer(x)
            x = F.gelu(norm_layer(x))
        return x, sensor_outputs


class BayesianInverseOperator(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_sensors: int,
        hidden_width: int = 48,
        modes: int = 8,
        depth: int = 4,
        sensor_strategy: str = "learned_softmax",
        sensor_temperature: float = 0.2,
        sensor_repulsion_scale: float = 0.02,
        sensor_sigma: float = 0.075,
        sensor_seed: int = 7,
        use_forcing_sensor_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = OperatorBackbone(
            grid_size=grid_size,
            num_sensors=num_sensors,
            hidden_width=hidden_width,
            modes=modes,
            depth=depth,
            sensor_strategy=sensor_strategy,
            sensor_temperature=sensor_temperature,
            sensor_repulsion_scale=sensor_repulsion_scale,
            sensor_sigma=sensor_sigma,
            sensor_seed=sensor_seed,
            use_forcing_sensor_fusion=use_forcing_sensor_fusion,
        )
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_width, hidden_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_width, 2, kernel_size=1),
        )

    @property
    def sensor_layer(self) -> BaseSensorLayer:
        return self.backbone.sensor_layer

    def forward(
        self,
        solution: torch.Tensor,
        forcing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x, sensor_outputs = self.backbone.encode(solution=solution, forcing=forcing)
        posterior = self.output_head(x)
        return {
            "posterior_mean": posterior[:, :1],
            "posterior_logvar": posterior[:, 1:].clamp(min=-6.0, max=3.0),
            **sensor_outputs,
        }


class DeterministicInverseOperator(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_sensors: int,
        hidden_width: int = 48,
        modes: int = 8,
        depth: int = 4,
        sensor_strategy: str = "learned_softmax",
        sensor_temperature: float = 0.2,
        sensor_repulsion_scale: float = 0.02,
        sensor_sigma: float = 0.075,
        sensor_seed: int = 7,
        use_forcing_sensor_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = OperatorBackbone(
            grid_size=grid_size,
            num_sensors=num_sensors,
            hidden_width=hidden_width,
            modes=modes,
            depth=depth,
            sensor_strategy=sensor_strategy,
            sensor_temperature=sensor_temperature,
            sensor_repulsion_scale=sensor_repulsion_scale,
            sensor_sigma=sensor_sigma,
            sensor_seed=sensor_seed,
            use_forcing_sensor_fusion=use_forcing_sensor_fusion,
        )
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_width, hidden_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_width, 1, kernel_size=1),
        )

    @property
    def sensor_layer(self) -> BaseSensorLayer:
        return self.backbone.sensor_layer

    def forward(
        self,
        solution: torch.Tensor,
        forcing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x, sensor_outputs = self.backbone.encode(solution=solution, forcing=forcing)
        mean = self.output_head(x)
        return {
            "posterior_mean": mean,
            "posterior_logvar": torch.full_like(mean, -6.0),
            **sensor_outputs,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetInverseModel(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_sensors: int,
        hidden_width: int = 32,
        sensor_strategy: str = "learned_softmax",
        sensor_temperature: float = 0.2,
        sensor_repulsion_scale: float = 0.02,
        sensor_sigma: float = 0.075,
        sensor_seed: int = 7,
        dropout: float = 0.0,
        use_forcing_sensor_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.use_forcing_sensor_fusion = use_forcing_sensor_fusion
        self.sensor_layer = build_sensor_layer(
            strategy=sensor_strategy,
            grid_size=grid_size,
            num_sensors=num_sensors,
            sensor_temperature=sensor_temperature,
            sensor_repulsion_scale=sensor_repulsion_scale,
            sensor_sigma=sensor_sigma,
            sensor_seed=sensor_seed,
        )
        feature_multiplier = 6 if use_forcing_sensor_fusion else 3
        in_channels = feature_multiplier * num_sensors + 4
        self.enc1 = ConvBlock(in_channels, hidden_width, dropout=dropout)
        self.enc2 = ConvBlock(hidden_width, hidden_width * 2, dropout=dropout)
        self.enc3 = ConvBlock(hidden_width * 2, hidden_width * 4, dropout=dropout)
        self.bottleneck = ConvBlock(hidden_width * 4, hidden_width * 4, dropout=dropout)
        self.up2 = nn.ConvTranspose2d(hidden_width * 4, hidden_width * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(hidden_width * 4, hidden_width * 2, dropout=dropout)
        self.up1 = nn.ConvTranspose2d(hidden_width * 2, hidden_width, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(hidden_width * 2, hidden_width, dropout=dropout)
        self.output_head = nn.Conv2d(hidden_width, 1, kernel_size=1)

    def encode_input(self, solution: torch.Tensor, forcing: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sensor_outputs = self.sensor_layer(
            solution,
            forcing=forcing,
            use_forcing_sensor_fusion=self.use_forcing_sensor_fusion,
        )
        features = sensor_outputs["features"]
        if forcing is None:
            forcing = torch.ones(
                features.shape[0],
                1,
                self.grid_size,
                self.grid_size,
                device=features.device,
                dtype=features.dtype,
            )
        coords = build_coordinate_grid(
            batch_size=features.shape[0],
            grid_size=self.grid_size,
            device=features.device,
            dtype=features.dtype,
        )
        return torch.cat((features, forcing, coords), dim=1), sensor_outputs

    def forward(self, solution: torch.Tensor, forcing: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x, sensor_outputs = self.encode_input(solution=solution, forcing=forcing)
        x1 = self.enc1(x)
        x2 = self.enc2(F.avg_pool2d(x1, kernel_size=2, ceil_mode=True))
        x3 = self.enc3(F.avg_pool2d(x2, kernel_size=2, ceil_mode=True))
        x4 = self.bottleneck(x3)
        y2 = self.up2(x4)
        y2 = F.interpolate(y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.dec2(torch.cat((y2, x2), dim=1))
        y1 = self.up1(y2)
        y1 = F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(torch.cat((y1, x1), dim=1))
        mean = self.output_head(y1)
        return {
            "posterior_mean": mean,
            "posterior_logvar": torch.full_like(mean, -6.0),
            **sensor_outputs,
        }


def build_model(
    model_name: str,
    grid_size: int,
    num_sensors: int,
    hidden_width: int,
    modes: int,
    depth: int,
    sensor_strategy: str,
    sensor_temperature: float,
    sensor_repulsion_scale: float,
    sensor_sigma: float,
    sensor_seed: int,
    dropout: float = 0.1,
    use_forcing_sensor_fusion: bool = False,
) -> nn.Module:
    if model_name == "bayesian_operator":
        return BayesianInverseOperator(
            grid_size=grid_size,
            num_sensors=num_sensors,
            hidden_width=hidden_width,
            modes=modes,
            depth=depth,
            sensor_strategy=sensor_strategy,
            sensor_temperature=sensor_temperature,
            sensor_repulsion_scale=sensor_repulsion_scale,
            sensor_sigma=sensor_sigma,
            sensor_seed=sensor_seed,
            use_forcing_sensor_fusion=use_forcing_sensor_fusion,
        )
    if model_name == "deterministic_operator":
        return DeterministicInverseOperator(
            grid_size=grid_size,
            num_sensors=num_sensors,
            hidden_width=hidden_width,
            modes=modes,
            depth=depth,
            sensor_strategy=sensor_strategy,
            sensor_temperature=sensor_temperature,
            sensor_repulsion_scale=sensor_repulsion_scale,
            sensor_sigma=sensor_sigma,
            sensor_seed=sensor_seed,
            use_forcing_sensor_fusion=use_forcing_sensor_fusion,
        )
    if model_name == "unet_deterministic":
        return UNetInverseModel(
            grid_size=grid_size,
            num_sensors=num_sensors,
            hidden_width=max(16, hidden_width // 2),
            sensor_strategy=sensor_strategy,
            sensor_temperature=sensor_temperature,
            sensor_repulsion_scale=sensor_repulsion_scale,
            sensor_sigma=sensor_sigma,
            sensor_seed=sensor_seed,
            dropout=dropout,
            use_forcing_sensor_fusion=use_forcing_sensor_fusion,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
