"""Explore Lunar Lander using the keyboard, fixed rules, or our neural network."""

import argparse
from pathlib import Path  # Handles the saved weights file.

import gymnasium as gym
import pygame
import torch  # Converts observations to tensors and samples neural network actions.
from gymnasium.envs.box2d.lunar_lander import heuristic
import config
from policy import PolicyNetwork, load_policy  # Load each saved model's own architecture.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    controller = parser.add_mutually_exclusive_group()  # Choose one automatic controller.
    controller.add_argument("--demo", action="store_true", help="Watch the built-in autopilot")
    controller.add_argument("--nn", action="store_true", help="Watch our neural network")
    parser.add_argument("--model", type=Path, help="Saved weights to load (requires --nn)")
    parser.add_argument("--greedy", action="store_true", help="Choose the most likely action (requires --nn)")
    args = parser.parse_args()
    if (args.model or args.greedy) and not args.nn:
        parser.error("--model and --greedy require --nn")
    if args.model and not args.model.is_file():
        parser.error(f"weights file does not exist: {args.model}")

    policy = None
    if args.nn:
        policy = load_policy(args.model) if args.model else PolicyNetwork()
    if policy is not None:
        policy.eval()  # Set the network to inference mode; this does not disable gradients.
    env = gym.make(config.ENV_ID, render_mode="human")
    try:
        observation, _ = env.reset()
        score = 0.0
        finished = False
        mode = "Manual"
        if args.demo:
            mode = "Autopilot"
        elif args.nn:
            mode = "Saved NN" if args.model else "Untrained NN"
        instructions = "Up: main engine | Left/Right: side engines | R: restart | Esc: quit"
        print(f"Lunar Lander — {mode}\n{instructions}")
        if args.nn and not args.model:
            print("The network has random weights; it is choosing actions without training.")
        elif args.model:
            print(f"Loaded weights from {args.model}; this window does not train them.")

        while True:
            restart = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_r:
                        restart = True

            if restart:
                observation, _ = env.reset()
                score, finished = 0.0, False

            if finished:
                # Keep the final scene visible until the user restarts or quits.
                env.render()
            else:
                keys = pygame.key.get_pressed()
                if args.nn:
                    with torch.no_grad():  # Choose actions without tracking gradients.
                        state = torch.tensor(observation, dtype=torch.float32)  # 8 inputs.
                        probabilities = policy(state)  # Calculate 4 action probabilities.
                        if args.greedy:
                            action = probabilities.argmax().item()  # Same choice as evaluation.
                        else:
                            distribution = torch.distributions.Categorical(probs=probabilities)
                            action = distribution.sample().item()  # Sample action 0, 1, 2, or 3.
                elif args.demo:
                    action = heuristic(env.unwrapped, observation)
                elif keys[pygame.K_UP]:
                    action = 2
                elif keys[pygame.K_LEFT]:
                    action = 1
                elif keys[pygame.K_RIGHT]:
                    action = 3
                else:
                    action = 0

                observation, reward, terminated, truncated, _ = env.step(action)
                score += reward
                finished = terminated or truncated
                if finished:
                    print(f"Episode ended. Score: {score:.1f}. Press R to restart.")

            status = "Episode ended — R to restart" if finished else instructions
            pygame.display.set_caption(f"Lunar Lander | {mode} | Score: {score:.1f} | {status}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
