"""Our first neural network: 8 state values in, 4 action probabilities out."""

import gymnasium as gym  # Provides the rocket simulation for the example below.
import torch  # Provides tensors (arrays of numbers) and automatic differentiation.
from torch import nn  # Provides the building blocks for neural networks.
import config  # All experiment defaults live in this one file.


class PolicyNetwork(nn.Module):  # A policy is a network that helps choose actions.
    def __init__(self, hidden_sizes=None, activation=None):
        super().__init__()  # Initializes PyTorch's tracking of layers and weights.
        self.hidden_sizes = tuple(config.HIDDEN_SIZES if hidden_sizes is None else hidden_sizes)
        self.activation = config.ACTIVATION if activation is None else activation
        if not self.hidden_sizes or any(type(size) is not int or size < 1 for size in self.hidden_sizes):
            raise ValueError("HIDDEN_SIZES must contain positive integers, such as (64, 64)")
        if self.activation not in ("relu", "tanh"):
            raise ValueError("ACTIVATION must be 'relu' or 'tanh'")
        activation_layer = nn.ReLU if self.activation == "relu" else nn.Tanh
        layers = []  # Build as many hidden layers as the config requests.
        previous_size = config.INPUT_SIZE  # First layer receives the eight state values.
        for size in self.hidden_sizes:
            layers.append(nn.Linear(previous_size, size))  # Learn combinations of inputs.
            layers.append(activation_layer())  # Add the chosen nonlinear transformation.
            previous_size = size  # This layer's outputs become the next layer's inputs.
        layers.append(nn.Linear(previous_size, config.ACTION_COUNT))  # Four raw action scores.
        layers.append(nn.Softmax(dim=-1))  # Convert scores into action probabilities.
        self.layers = nn.Sequential(*layers)  # Apply the assembled layers in order.

    def forward(self, state):  # PyTorch calls this when we write policy(state).
        return self.layers(state)  # Return the probabilities calculated by the layers.


def save_policy(policy, path):
    """Save the architecture with the learned weights so experiments stay loadable."""
    torch.save({
        "state_dict": policy.state_dict(),  # Learned weight and bias tensors.
        "hidden_sizes": list(policy.hidden_sizes),  # Layer widths used for this model.
        "activation": policy.activation,  # ReLU and Tanh have different behavior.
    }, path)


def load_policy(path):
    """Load new checkpoints and our original weights-only ReLU checkpoints."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
        policy = PolicyNetwork(checkpoint["hidden_sizes"], checkpoint["activation"])
    else:
        weights = checkpoint  # Older files contain only weight and bias tensors.
        keys = sorted((key for key in weights if key.endswith(".weight")),
                      key=lambda key: int(key.split(".")[1]))
        hidden_sizes = [weights[key].shape[0] for key in keys[:-1]]
        policy = PolicyNetwork(hidden_sizes, "relu")  # Original checkpoints used ReLU.
    policy.load_state_dict(weights)  # Restore values into the matching architecture.
    policy.eval()  # Ready for inference; loading never trains the network.
    return policy


if __name__ == "__main__":  # Run this example only when this file is executed directly.
    torch.manual_seed(config.SEED)  # Make initial weights and sampling reproducible.
    policy = PolicyNetwork()  # Create the network with randomly initialized weights.
    env = gym.make(config.ENV_ID)  # Create an environment without opening a window.
    try:  # Ensure we close the environment even if something goes wrong.
        observation, _ = env.reset(seed=config.SEED)  # Get the initial state; ignore extra info.
        state = torch.tensor(observation, dtype=torch.float32)  # Convert it to a tensor.
        with torch.no_grad():  # Skip gradient tracking while just inspecting the output.
            probabilities = policy(state)  # Feed the 8 numbers through the network.
        distribution = torch.distributions.Categorical(probs=probabilities)  # Four choices.
        action = distribution.sample().item()  # Sample an action; convert it to an int.
        print("State:", observation)  # Show the 8 input values from the environment.
        print("Action probabilities:", probabilities.tolist())  # Show the 4 outputs.
        print("Chosen action:", action)  # 0: off, 1: left, 2: main, 3: right.
        print("These weights are untrained; this example does not update them.")
    finally:  # Always release the environment's resources.
        env.close()
