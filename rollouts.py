"""Collect complete episodes in persistent worker processes, then learn in train.py."""

import atexit
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import numpy as np
import torch

import config
from policy import PolicyNetwork


def collect_episode(env, policy, seed):
    """Play one complete attempt with fixed weights; return aligned experience lists."""
    observation, _ = env.reset(seed=seed)  # Set this episode's initial conditions.
    states, actions, rewards = [], [], []
    # Each episode gets its own sampling seed, independent of which worker runs it.
    # Restore the caller's RNG afterward so collection doesn't change its random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        finished = False
        while not finished:
            state = torch.tensor(observation, dtype=torch.float32)
            with torch.no_grad():  # Workers collect experience; they never learn.
                probabilities = policy(state)
                distribution = torch.distributions.Categorical(probs=probabilities)
                action = distribution.sample().item()
            observation, reward, terminated, truncated, _ = env.step(action)
            states.append(state)  # Inputs before the action.
            actions.append(action)  # Action chosen for those inputs.
            rewards.append(float(reward))  # Reward after that action.
            finished = terminated or truncated
    return states, actions, rewards


def _initialize_worker(env_id, hidden_sizes, activation):
    """Create one environment and policy per process, reused for the whole run."""
    global _worker_env, _worker_policy
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Let the parent handle Ctrl+C and saving.
    torch.set_num_threads(1)  # Avoid each worker creating competing CPU thread pools.
    _worker_env = gym.make(env_id)
    _worker_policy = PolicyNetwork(hidden_sizes, activation)
    atexit.register(_worker_env.close)


def _collect_worker_batch(weights, episodes):
    """Load this batch's policy snapshot and return complete episodes as NumPy arrays."""
    _worker_policy.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    results = []
    for index, seed in episodes:
        states, actions, rewards = collect_episode(_worker_env, _worker_policy, seed)
        # Send one compact array per episode rather than thousands of individual tensors.
        results.append((index, np.stack([state.numpy() for state in states]), actions, rewards))
    return results


class EpisodeCollector:
    """Parallelize collection only; all gradient updates remain in the parent process."""

    def __init__(self, policy, workers):
        self.workers = workers
        self.env = None
        self.pool = None
        if workers == 1:
            self.env = gym.make(config.ENV_ID)  # Keep a simple sequential comparison mode.
        else:
            self.pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),  # Works safely on macOS.
                initializer=_initialize_worker,
                initargs=(config.ENV_ID, policy.hidden_sizes, policy.activation),
            )

    def collect(self, policy, seeds):
        """Return exactly these episodes, in order, all using the same policy weights."""
        if self.pool is None:
            return [collect_episode(self.env, policy, seed) for seed in seeds]
        weights = {name: value.detach().cpu().numpy().copy() for name, value in policy.state_dict().items()}
        episodes = list(enumerate(seeds))
        futures = [self.pool.submit(_collect_worker_batch, weights, episodes[index::self.workers])
                   for index in range(min(self.workers, len(episodes)))]
        results = [episode for future in futures for episode in future.result()]
        results.sort(key=lambda episode: episode[0])  # Preserve the original episode ordering.
        return [(list(torch.from_numpy(states).unbind()), actions, rewards)
                for _, states, actions, rewards in results]

    def close(self):
        """Finish pending short rollouts and release every worker when training ends."""
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.env is not None:
            self.env.close()
