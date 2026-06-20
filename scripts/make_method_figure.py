from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a paper-style method overview figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/method_overview.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def add_box(ax, xy: tuple[float, float], width: float, height: float, title: str, body: str, facecolor: str) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor="#1f2933",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + 0.02, xy[1] + height - 0.05, title, fontsize=14, fontweight="bold", color="#102a43")
    ax.text(xy[0] + 0.02, xy[1] + height - 0.12, body, fontsize=10.5, color="#243b53", va="top")


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], text: str | None = None) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.0,
        color="#486581",
    )
    ax.add_patch(arrow)
    if text:
        mid_x = 0.5 * (start[0] + end[0])
        mid_y = 0.5 * (start[1] + end[1])
        ax.text(mid_x, mid_y + 0.03, text, ha="center", va="bottom", fontsize=10, color="#334e68")


def add_mini_field(ax, extent: list[float], seed: int, cmap: str) -> None:
    rng = np.random.default_rng(seed)
    grid = rng.normal(size=(20, 20))
    grid = (grid + np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=1)) / 3.0
    ax.imshow(grid, extent=extent, origin="lower", cmap=cmap, alpha=0.95, zorder=0)
    ax.add_patch(
        FancyBboxPatch(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            boxstyle="round,pad=0.0,rounding_size=0.015",
            linewidth=1.2,
            edgecolor="#243b53",
            facecolor="none",
        )
    )


def add_sensor_dots(ax, base_x: float, base_y: float, width: float, height: float) -> None:
    sensors = np.array(
        [
            [0.18, 0.24],
            [0.30, 0.66],
            [0.55, 0.42],
            [0.67, 0.77],
            [0.82, 0.29],
            [0.74, 0.58],
        ]
    )
    xs = base_x + width * sensors[:, 0]
    ys = base_y + height * sensors[:, 1]
    ax.scatter(xs, ys, s=28, color="#d64545", zorder=5)


def main() -> None:
    args = parse_args()

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#f8fbff")
    ax.set_facecolor("#f8fbff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.04, 0.62),
        0.20,
        0.26,
        "Coefficient Field",
        "Unknown log-permeability\nfield a(x) to recover.",
        "#e0fbfc",
    )
    add_mini_field(ax, [0.065, 0.215, 0.655, 0.835], seed=11, cmap="viridis")

    add_box(
        ax,
        (0.04, 0.20),
        0.20,
        0.26,
        "Forcing Field",
        "Constant or multisource\nforcing f(x).",
        "#fff3bf",
    )
    add_mini_field(ax, [0.065, 0.215, 0.235, 0.415], seed=23, cmap="magma")

    add_box(
        ax,
        (0.31, 0.39),
        0.22,
        0.32,
        "Forward Darcy Solve",
        "Finite-difference PDE solve\nproduces pressure field u(x).",
        "#d9e2ec",
    )
    add_mini_field(ax, [0.345, 0.495, 0.445, 0.625], seed=31, cmap="cividis")

    add_box(
        ax,
        (0.58, 0.58),
        0.18,
        0.22,
        "Sensor Design",
        "Differentiable Gaussian-random\nsensor maps with repulsion.",
        "#fde2e4",
    )
    add_sensor_dots(ax, 0.61, 0.615, 0.12, 0.12)

    add_box(
        ax,
        (0.58, 0.25),
        0.18,
        0.18,
        "Sparse Observations",
        "Sensor responses y_S and\ncoordinates S.",
        "#e4c1f9",
    )

    add_box(
        ax,
        (0.80, 0.36),
        0.17,
        0.34,
        "Bayesian Inverse Operator",
        "Fourier-style operator backbone\noutputs posterior mean and log-variance.",
        "#c3f0ca",
    )

    add_box(
        ax,
        (0.25, 0.05),
        0.62,
        0.13,
        "Training Objective",
        "Coefficient MSE + log-MSE + Gaussian NLL + physics residual + sensor diversity/repulsion penalties.",
        "#f0f4f8",
    )

    add_arrow(ax, (0.24, 0.75), (0.31, 0.60))
    add_arrow(ax, (0.24, 0.33), (0.31, 0.48))
    add_arrow(ax, (0.53, 0.55), (0.58, 0.66), "pressure field")
    add_arrow(ax, (0.53, 0.49), (0.58, 0.34), "sampled at S")
    add_arrow(ax, (0.76, 0.66), (0.80, 0.58), "sensor layout")
    add_arrow(ax, (0.76, 0.34), (0.80, 0.45), "observations")
    add_arrow(ax, (0.60, 0.18), (0.60, 0.25))
    add_arrow(ax, (0.88, 0.36), (0.88, 0.18))

    ax.text(0.885, 0.73, "Posterior over log a(x)", ha="center", fontsize=11, color="#102a43", fontweight="bold")
    ax.text(0.885, 0.27, "Mean, variance,\nand calibration metrics", ha="center", fontsize=10.5, color="#334e68")
    ax.text(0.50, 0.94, "Bayesian inverse neural operator with learned sparse sensor design", ha="center", fontsize=17, fontweight="bold", color="#102a43")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
