# Working on Rocket Land

This is an educational Python project using Gymnasium LunarLander-v3 and a PyTorch REINFORCE policy. Preserve the readable, commented learning flow.

## Setup and validation

Run commands from the repository root:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run policy.py
```

For changes affecting the training flow, run a short smoke experiment with a new, unused output path:

```sh
uv run train.py --episodes 16 --batch-size 8 --workers 2 --output models/agent-smoke.pt --no-open-plot
```

Do not start a long training run unless the user requests one. The smoke run verifies execution, not learning performance. Use `--no-open-plot` when there is no desktop. Interactive `main.py` requires a graphical display; use `--demo` for the fixed-rule controller or `--nn --model PATH --greedy` for a saved network.

## Project conventions

- Read README.md for commands and docs/learning-guide.md for the math.
- All configurable experiment defaults belong in config.py. CLI overrides are per run.
- Every train.py invocation starts a fresh network; there is no resume flag.
- Keep the user's tuning choices unless the task asks to change them.
- Use unique output filenames; do not overwrite existing models, CSVs, charts, or visualizations without authorization.
- models/ and visualizations/ are ignored generated outputs. Curated, explicitly published experiments live in results/.
- Record effective settings, exact commands, code revision, and evaluation scores for requested experiments. Never infer historical settings from the current config.
- Training batch reward and separate greedy evaluation are different measurements. Policy-gradient loss does not need to decrease across batches.
- Each batch uses one fixed policy for collection and one parent-process update. Compute returns separately for each episode. Preserve per-episode seeds and ordered results across worker counts.
- Checkpoints store weights and architecture. Load with policy.load_policy; preserve legacy weights-only ReLU compatibility.
- Keep model paths consistent between training, viewing, heatmaps, and export.
