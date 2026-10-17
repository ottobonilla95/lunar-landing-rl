"""Export a saved policy, including its layers, so Netron can display a diagram."""

import argparse  # Read the model and output paths from the command line.
from pathlib import Path  # Work with filesystem paths.

import torch  # Load the weights and record the network's operations.

import config
from policy import load_policy  # Restore the saved model's own architecture.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=config.MODEL_TO_INSPECT)
    parser.add_argument("--output", type=Path, default=config.DIAGRAM_OUTPUT)
    args = parser.parse_args()

    policy = load_policy(args.model)  # Load its architecture and learned values together.
    example_state = torch.zeros(1, config.INPUT_SIZE)  # One observation with eight values.
    with torch.no_grad():  # Exporting needs no gradients or training.
        diagram = torch.export.export(policy, (example_state,))  # Record the computation graph.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(diagram, args.output)  # Save layers, operations, and weights together.
    print(f"Diagram model saved to {args.output}")


if __name__ == "__main__":
    main()
