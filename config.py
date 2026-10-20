"""Edit the experiment settings here, then run: uv run train.py."""

from pathlib import Path

# Network: try (32, 32), (128, 128), or (64,) for one hidden layer.
HIDDEN_SIZES = (128, 128)  # Number of neurons in each hidden layer.
ACTIVATION = "relu"  # Choose "relu" or "tanh" for the hidden layers.

# Training: each run starts a fresh network. CLI arguments override these defaults.
EPISODES = 5000  # Total landing attempts.
BATCH_SIZE = 8  # Complete attempts collected before one weight update.
PARALLEL_WORKERS = 8  # Benchmarked fastest here; set 1 for sequential collection.
LEARNING_RATE = 0.003  # Adam's step size; try 0.0003 as another experiment.
GAMMA = 0.99  # Discount factor; try 0.98 or 0.995.
SEED = 42  # Controls initial weights, action sampling, and training starts.
TORCH_THREADS = 1  # CPU threads used by PyTorch.
MAX_GRAD_NORM = 1.0  # Clip large gradients before Adam updates the weights.
NORMALIZATION_EPSILON = 1e-8  # Prevent division by zero when scaling returns.
OUTPUT = Path("models/policy.pt")  # Rerunning with this path replaces its outputs.

# Evaluation: use the same test starts before and after training.
EVALUATION_EPISODES = 10  # Number of greedy test attempts.
EVALUATION_SEED = 10_000  # Separate from the training starts.

# Charts: training saves the PNG and opens it in the computer's image viewer.
AUTO_OPEN_PLOT = True  # Set False to save the chart without opening it.
PLOT_SMOOTHING_WINDOW = 10  # Updates in the trailing average on the chart.

# Standalone visualization tools: choose which saved model to inspect.
MODEL_TO_INSPECT = Path("models/long_run.pt")
DIAGRAM_OUTPUT = Path("models/policy-diagram.pt2")
HEATMAP_OUTPUT = Path("visualizations/weight-heatmaps.png")

# These describe Lunar Lander's fixed interface, rather than tuning parameters.
ENV_ID = "LunarLander-v3"
INPUT_SIZE = 8
ACTION_COUNT = 4
INPUT_NAMES = ("x", "y", "x velocity", "y velocity", "angle", "angular velocity", "left leg", "right leg")
ACTION_NAMES = ("0 · engines off", "1 · left engine", "2 · main engine", "3 · right engine")
