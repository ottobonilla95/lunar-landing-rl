# Commands and project reference

[Back to the project](../README.md) · [Learning guide](learning-guide.md) · [Experiment gallery](../results/README.md)

Run these commands from the repository root after `uv sync --locked`. Choose new output names for each experiment.

## Use Python directly

If you prefer `python main.py`, activate the environment after `uv sync --locked`:

```sh
# macOS / Linux
source .venv/bin/activate
python main.py --demo
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. The scripts are ordinary Python; `uv run` selects the correct environment for you.

## Experiment settings

Edit **[config.py](../config.py)** to keep settings in one place. These are the checked-in defaults; local edits and earlier experiments may differ:

| Setting | Current value | Meaning |
| --- | --- | --- |
| `HIDDEN_SIZES` | `(128, 128)` | Two hidden layers, 128 neurons each |
| `ACTIVATION` | `"relu"` | Hidden activation; `"tanh"` is also supported |
| `EPISODES` | `7000` | Complete landing attempts |
| `BATCH_SIZE` | `8` | Attempts collected before one update |
| `PARALLEL_WORKERS` | `8` | CPU processes collecting episodes |
| `LEARNING_RATE` | `0.004` | Adam's learning rate |
| `GAMMA` | `0.99` | Discount applied per future step |
| `SEED` | `42` | Initial weights, training starts, and sampled actions |
| `TORCH_THREADS` | `1` | PyTorch CPU threads per process |
| `MAX_GRAD_NORM` | `1.0` | Gradient clipping limit |
| `EVALUATION_EPISODES` | `10` | Greedy test attempts before and after training |
| `EVALUATION_SEED` | `10000` | Starting seed for the fixed evaluation set |
| `OUTPUT` | `models/policy.pt` | Default output path |
| `AUTO_OPEN_PLOT` | `True` | Open the chart after training |

For example, `(64, 64)`, `(32, 32)`, or `(64,)` select other architectures. Saved models remember their own architecture even if you later change the config. Chart smoothing and visualization output paths are also configured there.

Command-line options override matching defaults for a single run:

```sh
uv run train.py --help
uv run train.py --episodes 5000 --batch-size 8 --workers 4 --learning-rate 0.003 --gamma 0.99 --seed 43 --output models/seed43.pt
```

To collect sequentially, use `--workers 1`. Parallel workers collect complete episodes with unchanged copies of the current policy. The parent waits for the batch and updates the network once. Worker count does not change the batch size or number of updates; it is capped by the batch size and episode count. The current implementation uses per-episode random seeds and preserves episode order across worker counts. This differs from older versions that used one shared random sequence.

## How the files connect

```text
config.py → policy.py → train.py → models/<run>.pt → main.py --nn --model ...
                           ↑
                       rollouts.py
                           ↓
                  CSV → plot_training.py → PNG

saved model → plot_weights.py → weight heatmaps
saved model → export_model.py → Netron diagram
```

| File | Responsibility |
| --- | --- |
| `config.py` | Experiment settings and the environment interface |
| `policy.py` | Network definition, sampling example, checkpoint saving/loading |
| `main.py` | Interactive manual, fixed-rule, or neural network controller |
| `rollouts.py` | Collect complete episodes sequentially or in reusable processes |
| `train.py` | Reward-to-go, policy update, evaluation, and training loop |
| `plot_training.py` | Plot recorded loss and training reward |
| `plot_architecture.py` | Draw layers, neuron counts, activations, and parameter counts |
| `plot_weights.py` | Plot each linear layer's connection weights |
| `export_model.py` | Export the computation graph for Netron |
| `tests/test_training.py` | Learning math, checkpoints, and collection checks |

## Inspect a trained model

Use the path of a model you have actually trained:

```sh
# Regenerate a training chart from its CSV.
uv run plot_training.py models/ep5000-lr0003.csv

# Create PNG and SVG heatmaps; each cell is one connection weight.
uv run plot_weights.py --model models/ep5000-lr0003.pt --output visualizations/ep5000-weights.png

# Draw the saved model's own architecture as PNG and SVG.
uv run plot_architecture.py --model models/ep5000-lr0003.pt --output visualizations/ep5000-architecture.png

# Export the layer diagram, then view it with Netron.
uv run export_model.py --model models/ep5000-lr0003.pt --output models/ep5000-diagram.pt2
uvx netron models/ep5000-diagram.pt2 --port 8080
```

Open [localhost:8080](http://127.0.0.1:8080/) while Netron runs. Heatmap colors describe parameter values, not landing quality. Netron is an optional viewer, fetched separately by `uvx`.

## Regenerate the repository's network diagram

```sh
uv run plot_architecture.py --output docs/images/policy-architecture.png
```

This reads the current architecture from `config.py` and saves PNG and SVG. Use `--model PATH` to draw a saved checkpoint's own architecture instead.

## Train with a coding agent

Once setup is complete, copy this follow-up message when you want to start training:

> In lunar-landing-rl, read AGENTS.md and README.md. Train a fresh policy for 5,000 episodes with batch size 8, learning rate 0.003, and gamma 0.99. Keep the architecture from config.py and report it. Choose a new unused output filename under models/ so existing results are preserved. Record the effective config, exact command, console output, CSV, chart, and initial/final greedy evaluation scores. Use --no-open-plot without a desktop. After training, tell me the path and the matching uv run main.py --nn --model PATH --greedy command. Do not claim the training reward is a separate evaluation score.

## Check the implementation

```sh
uv run python -m unittest discover -s tests -v
```

The tests cover discounted returns, update direction, equal-return stability, checkpoint compatibility, output/config behavior, episode termination, and matching sequential/parallel collection. A short training run checks the plumbing; it does not measure long-run learning quality.
