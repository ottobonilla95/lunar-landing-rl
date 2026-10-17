"""Plot a training progress CSV; also called automatically at the end of training."""

import argparse
import csv
from pathlib import Path
import os
import subprocess
import sys

import config

import matplotlib

matplotlib.use("Agg")  # Save a plot even when training has no graphical window.
import matplotlib.pyplot as plt


def plot_training(csv_path):
    """Save loss and reward curves beside the CSV; return the image path."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as log_file:
        rows = list(csv.DictReader(log_file))  # Each row represents one completed update.
    if not rows:
        print("No completed updates yet; there is no training curve to plot.")
        return None

    episodes = [int(row["episodes"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    scores = [float(row["batch_mean_score"]) for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    if type(config.PLOT_SMOOTHING_WINDOW) is not int or config.PLOT_SMOOTHING_WINDOW < 1:
        raise ValueError("PLOT_SMOOTHING_WINDOW must be a positive integer")
    window = min(config.PLOT_SMOOTHING_WINDOW, len(rows))  # Reveal trends in noisy batches.
    for axis, values, title, label, color in zip(
        axes, [losses, scores],
        ["Policy-gradient loss", "Training reward · sampled actions"],
        ["Loss", "Mean episode score in each batch"], ["#7655aa", "#167d8d"],
    ):
        axis.plot(episodes, values, color=color, alpha=0.35, linewidth=1, marker=".", label="Each update")
        smoothed = [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
                    for i in range(len(values))]
        axis.plot(episodes, smoothed, color=color, linewidth=2, label=f"Trailing mean · up to {window} updates")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=9)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Completed training episodes")
    figure.suptitle(f"Lunar Lander training · {csv_path.stem}", fontsize=16)
    figure.supxlabel("Loss need not decrease or reach zero. Use rewards and separate evaluation to judge learning.", fontsize=10)
    output_path = csv_path.with_suffix(".png")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def open_plot(path):
    """Open the saved PNG in the operating system's image viewer, without blocking."""
    path = Path(path).resolve()
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        else:
            command = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.run([command, str(path)], check=True)  # macOS uses Preview by default.
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not open chart automatically ({error}). Open it manually: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Progress CSV written by train.py")
    args = parser.parse_args()
    output = plot_training(args.csv_path)
    if output is not None:
        print(f"Saved training plot: {output.resolve()}")
