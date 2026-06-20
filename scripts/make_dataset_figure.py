from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a representative dataset sample.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Dataset Example")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset)
    coefficient = data[f"{args.split}_coefficients"][args.index]
    solution = data[f"{args.split}_solutions"][args.index]
    forcing = data[f"{args.split}_forcings"][args.index]
    task_name = str(data["task"]) if "task" in data else "unknown"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.patch.set_facecolor("#f8fbff")
    cmaps = ["viridis", "cividis", "magma"]
    fields = [coefficient, solution, forcing]
    labels = ["Coefficient", "Pressure solution", "Forcing"]

    for ax, field, cmap, label in zip(axes, fields, cmaps, labels):
        image = ax.imshow(field, origin="lower", cmap=cmap)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{args.title}\n{task_name} | {args.split} sample #{args.index}", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
