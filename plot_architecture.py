"""Draw the policy's layers, neuron counts, activations, and parameter counts."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Generate files without opening a desktop window.
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import torch

import config
from policy import PolicyNetwork, load_policy


def plot_architecture(policy, output):
    """Draw an architecture summary; abbreviated circles represent wide layers."""
    linear_layers = [layer for layer in policy.layers if isinstance(layer, torch.nn.Linear)]
    activation = "ReLU" if policy.activation == "relu" else "Tanh"
    activation_formula = "max(0, z)" if policy.activation == "relu" else "tanh(z)"
    colors = {"ink": "#17283D", "muted": "#52647A", "line": "#CED9E4"}
    stages = [("OBSERVATION", config.INPUT_SIZE, "#137F87")]
    stages += [(f"HIDDEN LAYER {i + 1}", size, "#7355B5")
               for i, size in enumerate(policy.hidden_sizes)]
    stages += [("ACTION SCORES", config.ACTION_COUNT, "#B36A21"),
               ("ACTION PROBABILITIES", config.ACTION_COUNT, "#137F87")]
    width, gap = 3.55, 0.65
    canvas_width = 0.6 + len(stages) * (width + gap)
    figure, axis = plt.subplots(figsize=(canvas_width, 10), facecolor="#F4F7FB")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axis.set(xlim=(0, canvas_width), ylim=(0, 10))
    axis.axis("off")
    plt.rcParams["svg.fonttype"] = "none"  # Keep vector labels selectable.

    axis.text(0.6, 9.25, "Inside our Lunar Lander neural network",
              fontsize=27, weight="bold", color=colors["ink"])
    sizes = [config.INPUT_SIZE, *policy.hidden_sizes, config.ACTION_COUNT]
    axis.text(0.6, 8.72, " → ".join(map(str, sizes)) +
              f"   ·   {activation} hidden activations   ·   Softmax output",
              fontsize=16, color=colors["muted"])
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    axis.text(0.6, 8.17, f"{parameter_count:,} trainable parameters (weights + biases)",
              fontsize=13, weight="bold", color=colors["ink"])

    for index, (title, size, accent) in enumerate(stages):
        left = 0.6 + index * (width + gap)
        center = left + width / 2
        axis.add_patch(FancyBboxPatch((left, 2.55), width, 4.95,
                       boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor="white", edgecolor=colors["line"], linewidth=1.2))
        axis.text(center, 7.08, title, ha="center", fontsize=11,
                  weight="bold", color=accent)
        noun = "state values" if index == 0 else "neurons"
        if index == len(stages) - 2:
            noun = "logits"
        elif index == len(stages) - 1:
            noun = "probabilities"
        axis.text(center, 6.62, f"{size} {noun}", ha="center", fontsize=21,
                  weight="bold", color=colors["ink"])

        if index == 0:
            labels = config.INPUT_NAMES
        elif index == len(stages) - 2:
            labels = [f"z{i}  ·  raw score" for i in range(size)]
        elif index == len(stages) - 1:
            labels = [f"p{i}  ·  {name.split(' · ', 1)[-1]}"
                      for i, name in enumerate(config.ACTION_NAMES)]
        elif size <= 8:
            labels = [f"neuron {i + 1}" for i in range(size)]
        else:
            labels = ["neuron 1", "neuron 2", "neuron 3", "…",
                      f"neuron {size - 2}", f"neuron {size - 1}", f"neuron {size}"]

        # Show every input/output, but abbreviate large hidden layers.
        spacing = 0.34 if len(labels) > 6 else 0.53
        top = 4.75 + spacing * (len(labels) - 1) / 2
        for row, label in enumerate(labels):
            y = top - row * spacing
            if label != "…":
                axis.add_patch(Circle((left + 0.43, y), 0.09,
                               facecolor=accent, alpha=0.85, edgecolor="none"))
            axis.text(left + 0.67, y, label, va="center", fontsize=11,
                      color=colors["ink"])

        if 1 <= index <= len(policy.hidden_sizes):
            layer = linear_layers[index - 1]
            axis.text(center, 3.13, f"{activation}(z) = {activation_formula}",
                      ha="center", fontsize=12, weight="bold", color=accent)
        elif index == len(stages) - 2:
            layer = linear_layers[-1]
            axis.text(center, 3.13, "Unrestricted real numbers", ha="center",
                      fontsize=11, color=accent)
        elif index == len(stages) - 1:
            layer = None
            axis.text(center, 3.13, "Softmax · probabilities sum to 1", ha="center",
                      fontsize=11, weight="bold", color=accent)
        else:
            layer = None
            axis.text(center, 3.13, "From env.reset() / env.step()", ha="center",
                      fontsize=11, color=accent)

        if layer is not None:
            count = sum(parameter.numel() for parameter in layer.parameters())
            axis.text(center, 2.78, f"Linear({layer.in_features}, {layer.out_features})"
                      f" · {count:,} parameters", ha="center", fontsize=10,
                      color=colors["muted"])

        if index < len(stages) - 1:
            axis.add_patch(FancyArrowPatch((left + width + 0.08, 4.9),
                           (left + width + gap - 0.08, 4.9),
                           arrowstyle="-|>", mutation_scale=17, linewidth=1.8,
                           color=colors["muted"]))

    axis.text(0.6, 1.8, "One observation → one probability for each action",
              fontsize=16, weight="bold", color=colors["ink"])
    axis.text(0.6, 1.32, "Training: sample an action from these probabilities."
              "     Greedy evaluation: choose the largest probability.",
              fontsize=12, color=colors["muted"])
    axis.text(0.6, 0.81, "Each Linear layer is fully connected and learns weights and biases."
              f" {activation} and Softmax have no trainable parameters.",
              fontsize=12, color=colors["muted"])
    axis.text(0.6, 0.37, "Neuron circles are schematic; hidden layers are abbreviated."
              " Arrows carry complete vectors. This shows architecture, not learned weight values.",
              fontsize=11, color=colors["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140, facecolor=figure.get_facecolor())
    svg_path = output.with_suffix(".svg")
    figure.savefig(svg_path, facecolor=figure.get_facecolor())
    # Matplotlib includes trailing spaces in SVG paths; keep generated diffs clean.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(figure)
    return parameter_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Read a saved model's own architecture")
    parser.add_argument("--output", type=Path,
                        default=Path("visualizations/policy-architecture.png"))
    args = parser.parse_args()
    policy = load_policy(args.model) if args.model else PolicyNetwork()
    count = plot_architecture(policy, args.output)
    print(f"Saved {args.output} and {args.output.with_suffix('.svg')} ({count:,} parameters)")


if __name__ == "__main__":
    main()
