"""Show the saved network's connection weights as three heatmaps.

Run with: uv run --with matplotlib plot_weights.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save an image without opening a separate application.
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

import config
from policy import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=config.MODEL_TO_INSPECT)
    parser.add_argument("--output", type=Path, default=config.HEATMAP_OUTPUT)
    args = parser.parse_args()

    policy = load_policy(args.model)  # The checkpoint decides its layer sizes and activation.
    matrices = [layer.weight.detach().numpy() for layer in policy.layers if isinstance(layer, torch.nn.Linear)]
    limit = max(float(np.abs(matrix).max()) for matrix in matrices)
    norm = TwoSlopeNorm(vmin=-max(limit, 1e-8), vcenter=0, vmax=max(limit, 1e-8))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 8), layout="constrained")
    titles = ["State → hidden layer 1"]
    titles.extend(f"Hidden layer {i} → hidden layer {i + 1}" for i in range(1, len(policy.hidden_sizes)))
    titles.append(f"Hidden layer {len(policy.hidden_sizes)} → action scores")

    for index, (axis, matrix, title) in enumerate(zip(axes, matrices, titles)):
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        rows, columns = matrix.shape
        axis.set_title(f"{title}\n{rows} × {columns} weights", fontweight="bold", pad=14)
        axis.set_xlabel("Source neuron (column)")
        axis.set_ylabel("Destination neuron (row)")
        if index == 0:
            axis.set_xticks(range(config.INPUT_SIZE), config.INPUT_NAMES, rotation=55, ha="right")
        else:
            ticks = np.unique(np.linspace(0, columns - 1, min(5, columns), dtype=int))
            axis.set_xticks(ticks, ticks + 1)
        if index == len(matrices) - 1:
            axis.set_yticks(range(config.ACTION_COUNT), config.ACTION_NAMES)
        else:
            ticks = np.unique(np.linspace(0, rows - 1, min(5, rows), dtype=int))
            axis.set_yticks(ticks, ticks + 1)
        for spine in axis.spines.values():
            spine.set_visible(False)

    colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Connection weight · same scale in every panel")
    figure.suptitle(
        f"Lunar Lander · saved neural network weights\n{args.model.name} · {sum(matrix.size for matrix in matrices):,} connections · blue: negative / white: zero / red: positive",
        fontsize=17, fontweight="bold",
    )
    figure.supxlabel(
        "One cell = one connection. Stronger color = larger weight magnitude.\n"
        "Colors show parameter values; landing quality is measured by evaluation scores. Biases are omitted.",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    figure.savefig(args.output.with_suffix(".svg"))
    plt.close(figure)
    print(f"Saved heatmaps to {args.output}")


if __name__ == "__main__":
    main()
