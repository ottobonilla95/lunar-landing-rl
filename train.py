"""Train our policy with REINFORCE: collect attempts, then update its weights."""

import argparse  # Reads options such as --episodes from the command line.
import csv  # Writes a small progress table we can inspect later.
from pathlib import Path  # Handles the output file and its parent folder.

import gymnasium as gym  # Provides the simulation and rewards.
import torch  # Provides tensors, gradients, and optimizers.
import config  # Change experiment settings in config.py.
from policy import PolicyNetwork, save_policy  # Same architecture as the simulation.
from plot_training import plot_training, open_plot  # Save and display the training history.
from rollouts import EpisodeCollector, collect_episode  # Collect the batch on multiple CPU cores.


def discounted_returns(rewards, gamma):
    """Calculate each action's reward plus discounted rewards that follow it."""
    returns = []  # One future-reward total for each step in this attempt.
    future_return = 0.0  # There are no rewards beyond the end of the attempt.
    for reward in reversed(rewards):  # Work backward so future rewards are already known.
        future_return = reward + gamma * future_return  # Immediate + discounted future.
        returns.append(future_return)  # Store the total for this step.
    return returns[::-1]  # Reverse the list back into chronological order.


def update_policy(policy, optimizer, states, actions, returns):
    """Use one fresh batch of experience for one gradient update."""
    states = torch.stack(states)  # Combine individual states into a [steps, 8] tensor.
    actions = torch.tensor(actions, dtype=torch.long)  # Action IDs must be integers.
    returns = torch.tensor(returns, dtype=torch.float32)  # Future-reward totals per step.
    weights = returns - returns.mean()  # Compare returns with the batch's average.
    weights = weights / (returns.std(unbiased=False) + config.NORMALIZATION_EPSILON)  # Avoid / 0.
    probabilities = policy(states)  # Recompute with gradient tracking ENABLED.
    distribution = torch.distributions.Categorical(probs=probabilities)
    log_probabilities = distribution.log_prob(actions)  # Log probability of each action TAKEN.
    loss = -(log_probabilities * weights).mean()  # Better returns encourage those actions.
    optimizer.zero_grad()  # Clear gradients left over from the previous update.
    loss.backward()  # Compute how every weight and bias affects this loss.
    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.MAX_GRAD_NORM)  # Limit large gradients.
    optimizer.step()  # Adam uses those gradients to change the weights and biases.
    return loss.item()  # Return a plain number for printing, not a computation graph.


def evaluate(policy, episodes=None, seed=None):
    """Measure scores on fixed test starts, using the most likely action each step."""
    episodes = config.EVALUATION_EPISODES if episodes is None else episodes
    seed = config.EVALUATION_SEED if seed is None else seed
    env = gym.make(config.ENV_ID)  # A separate simulation; no window is needed.
    scores = []  # Store each complete attempt's ordinary, undiscounted score.
    try:
        with torch.no_grad():  # Evaluation only measures; it never updates weights.
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + episode)  # Same test starts each time.
                score, finished = 0.0, False
                while not finished:
                    state = torch.tensor(observation, dtype=torch.float32)
                    action = policy(state).argmax().item()  # Pick the largest probability.
                    observation, reward, terminated, truncated, _ = env.step(action)
                    score += reward  # Sum actual rewards, without discounting them.
                    finished = terminated or truncated
                scores.append(score)
    finally:
        env.close()  # Release the test environment even if evaluation is interrupted.
    return sum(scores) / len(scores)  # Average over attempts, not just one lucky landing.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=config.EPISODES, help="Number of landing attempts")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Attempts before each update")
    parser.add_argument("--workers", type=int, default=config.PARALLEL_WORKERS, help="Parallel collection processes (1 for sequential)")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Seed for reproducibility")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Adam's step size")
    parser.add_argument("--gamma", type=float, default=config.GAMMA, help="Future reward discount (0 to 1)")
    parser.add_argument("--output", type=Path, default=config.OUTPUT, help="Weights file")
    parser.add_argument("--no-open-plot", dest="open_plot", action="store_false",
                        default=config.AUTO_OPEN_PLOT, help="Save the chart without opening it")
    args = parser.parse_args()
    if args.episodes < 1 or args.batch_size < 1 or args.workers < 1 or args.seed < 0:
        parser.error("episodes, batch-size and workers must be positive; seed must be nonnegative")
    if not 0 < args.learning_rate < float("inf") or not 0 <= args.gamma <= 1:
        parser.error("learning-rate must be finite and positive; gamma must be between 0 and 1")
    if config.TORCH_THREADS < 1 or config.EVALUATION_EPISODES < 1 or config.EVALUATION_SEED < 0:
        parser.error("TORCH_THREADS and EVALUATION_EPISODES must be positive; EVALUATION_SEED nonnegative")
    if not 0 < config.MAX_GRAD_NORM < float("inf") or not 0 < config.NORMALIZATION_EPSILON < float("inf"):
        parser.error("MAX_GRAD_NORM and NORMALIZATION_EPSILON must be finite and positive")
    if type(config.PLOT_SMOOTHING_WINDOW) is not int or config.PLOT_SMOOTHING_WINDOW < 1:
        parser.error("PLOT_SMOOTHING_WINDOW must be a positive integer")

    torch.set_num_threads(config.TORCH_THREADS)  # Default: one thread for this small network.
    torch.manual_seed(args.seed)  # Control the initial weights and action sampling.
    policy = PolicyNetwork()  # Start a new policy with random weights for this run.
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)  # Size of learning steps.
    gamma = args.gamma  # Default: next step counts 99% as much.
    args.output.parent.mkdir(parents=True, exist_ok=True)  # Create the output folder.
    print(f"Network: {config.INPUT_SIZE} → {policy.hidden_sizes} → {config.ACTION_COUNT} | {policy.activation}")
    print(f"Initial test mean ({config.EVALUATION_EPISODES} attempts): {evaluate(policy):.1f}")

    workers = min(args.workers, args.batch_size, args.episodes)  # Don't create idle workers.
    collector = EpisodeCollector(policy, workers)  # Reuse the workers throughout training.
    print(f"Collecting with {workers} worker(s) | {args.batch_size} episodes per update")
    try:
        with args.output.with_suffix(".csv").open("w", newline="") as log_file:
            writer = csv.writer(log_file)  # Save one progress row per weight update.
            writer.writerow(["episodes", "batch_mean_score", "loss"])
            for start in range(0, args.episodes, args.batch_size):  # One loop per batch.
                states, actions, returns, scores = [], [], [], []  # Fresh experience only.
                end = min(start + args.batch_size, args.episodes)  # Allow a smaller last batch.
                seeds = [args.seed + episode for episode in range(start, end)]
                episodes = collector.collect(policy, seeds)  # Wait for this entire batch.
                for episode_states, episode_actions, rewards in episodes:
                    states.extend(episode_states)  # Add the attempt's inputs to the batch.
                    actions.extend(episode_actions)  # Keep the corresponding chosen actions.
                    returns.extend(discounted_returns(rewards, gamma))  # No crossing attempts.
                    scores.append(sum(rewards))  # Ordinary score is separate from returns.
                loss = update_policy(policy, optimizer, states, actions, returns)  # Learn ONCE.
                mean_score = sum(scores) / len(scores)  # How this collection batch performed.
                writer.writerow([end, mean_score, loss])  # Record scores and the training loss.
                log_file.flush()  # Make progress visible on disk immediately.
                print(f"Episodes {end}/{args.episodes} | batch score {mean_score:.1f} | loss {loss:.4f}")
    except KeyboardInterrupt:
        print("\nStopped; saving the weights from the last completed update.")
    finally:
        collector.close()  # Always release environments and worker processes.

    save_policy(policy, args.output)  # Save weights and the architecture used in this run.
    print(f"Saved weights: {args.output}")
    plot_path = plot_training(args.output.with_suffix(".csv"))  # Plot every completed update.
    if plot_path is not None:
        print(f"Saved training plot: {plot_path.resolve()}")
    print(f"Final test mean (same {config.EVALUATION_EPISODES} attempts): {evaluate(policy):.1f}")
    if plot_path is not None and args.open_plot:
        open_plot(plot_path)  # Open the chart after training and final evaluation finish.


if __name__ == "__main__":  # Importing this file does not launch training.
    main()
