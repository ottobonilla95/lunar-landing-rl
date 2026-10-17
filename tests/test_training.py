"""Check the reward math and the direction of actual learning updates."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import contextlib
import io

import torch

import config
import train
from policy import PolicyNetwork, load_policy, save_policy
from train import collect_episode, discounted_returns, update_policy
from rollouts import EpisodeCollector


class TrainingTests(unittest.TestCase):
    def test_discounted_returns_include_only_future_rewards(self):
        self.assertEqual(discounted_returns([1.0, 2.0, 3.0], 0.5), [2.75, 3.5, 3.0])
        self.assertEqual(discounted_returns([1.0, 2.0, 3.0], 1.0), [6.0, 5.0, 3.0])
        self.assertEqual(discounted_returns([1.0, 2.0, 3.0], 0.0), [1.0, 2.0, 3.0])

    def test_update_encourages_better_return_and_discourages_worse_return(self):
        policy = PolicyNetwork()
        for parameter in policy.parameters():
            torch.nn.init.zeros_(parameter)
        state = torch.zeros(8)
        before = policy(state).detach().clone()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        loss = update_policy(policy, optimizer, [state, state], [0, 1], [3.0, -3.0])
        after = policy(state).detach()
        self.assertTrue(math.isfinite(loss))
        self.assertGreater(after[0].item(), before[0].item())
        self.assertLess(after[1].item(), before[1].item())
        self.assertAlmostEqual(after.sum().item(), 1.0, places=6)

    def test_identical_returns_do_not_produce_nan_or_update_weights(self):
        policy = PolicyNetwork()
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        loss = update_policy(policy, optimizer, [torch.zeros(8)], [0], [4.0])
        self.assertTrue(math.isfinite(loss))
        for old, new in zip(before, policy.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_checkpoint_preserves_action_probabilities(self):
        policy = PolicyNetwork()
        states = torch.randn(3, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save(policy.state_dict(), path)
            restored = PolicyNetwork()
            restored.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        self.assertTrue(torch.equal(policy(states), restored(states)))

    def test_saved_architecture_is_restored_after_config_changes(self):
        policy = PolicyNetwork((16, 8, 4), "tanh")
        states = torch.randn(3, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_policy(policy, path)
            with patch.object(config, "HIDDEN_SIZES", (128,)), patch.object(config, "ACTIVATION", "relu"):
                restored = load_policy(path)
        self.assertEqual(restored.hidden_sizes, (16, 8, 4))
        self.assertEqual(restored.activation, "tanh")
        self.assertTrue(torch.equal(policy(states), restored(states)))

    def test_original_weights_only_files_still_load_after_config_changes(self):
        policy = PolicyNetwork((64, 64), "relu")
        states = torch.randn(3, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(policy.state_dict(), path)
            with patch.object(config, "HIDDEN_SIZES", (32,)), patch.object(config, "ACTIVATION", "tanh"):
                restored = load_policy(path)
        self.assertEqual(restored.hidden_sizes, (64, 64))
        self.assertEqual(restored.activation, "relu")
        self.assertTrue(torch.equal(policy(states), restored(states)))

    def test_training_uses_config_and_opens_chart_after_final_evaluation(self):
        class TinyEnvironment:
            def reset(self, seed):
                return [0.0] * 8, {}

            def step(self, action):
                return [0.0] * 8, 1.0, True, False, {}

            def close(self):
                pass

        for extra_args, should_open in [([], True), (["--no-open-plot"], False)]:
            with self.subTest(open_chart=should_open), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "configured.pt"
                events = []
                settings = dict(EPISODES=2, BATCH_SIZE=1, HIDDEN_SIZES=(16,), ACTIVATION="tanh",
                                OUTPUT=output, AUTO_OPEN_PLOT=True, LEARNING_RATE=0.0003, GAMMA=0.98)
                with patch.multiple(config, **settings), patch("sys.argv", ["train.py", *extra_args]), \
                     patch("train.gym.make", return_value=TinyEnvironment()), \
                     patch("train.evaluate", side_effect=lambda _: events.append("evaluation") or 0.0), \
                     patch("train.open_plot", side_effect=lambda _: events.append("opened")) as opener, \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                self.assertTrue(output.with_suffix(".csv").is_file())
                self.assertGreater(output.with_suffix(".png").stat().st_size, 1000)
                self.assertEqual(load_policy(output).hidden_sizes, (16,))
                self.assertEqual(load_policy(output).activation, "tanh")
                if should_open:
                    opener.assert_called_once_with(output.with_suffix(".png"))
                    self.assertEqual(events, ["evaluation", "evaluation", "opened"])
                else:
                    opener.assert_not_called()

    def test_collection_stops_on_termination_or_truncation(self):
        class OneStepEnvironment:
            def __init__(self, truncated):
                self.truncated = truncated
                self.steps = 0

            def reset(self, seed):
                return [0.0] * 8, {}

            def step(self, action):
                self.steps += 1
                if self.steps > 1:
                    raise AssertionError("Stepped after episode ended")
                return [1.0] * 8, 5.0, not self.truncated, self.truncated, {}

        for truncated in [False, True]:
            with self.subTest(truncated=truncated):
                env = OneStepEnvironment(truncated)
                states, actions, rewards = collect_episode(env, PolicyNetwork(), seed=42)
                self.assertEqual(len(states), 1)
                self.assertTrue(torch.equal(states[0], torch.zeros(8)))
                self.assertIn(actions[0], range(4))
                self.assertEqual(rewards, [5.0])

    def test_parallel_collection_matches_sequential_and_handles_small_batches(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        policy = PolicyNetwork((16,), "tanh")
        serial = EpisodeCollector(policy, 1)
        parallel = EpisodeCollector(policy, 2)
        try:
            # Shuffle seeds to check original ordering, and include a partial final batch.
            for seeds in ([52, 50, 51], [60]):
                expected = serial.collect(policy, seeds)
                actual = parallel.collect(policy, seeds)
                self.assertEqual(len(actual), len(seeds))
                for (old_states, old_actions, old_rewards), (states, actions, rewards) in zip(expected, actual):
                    self.assertTrue(torch.equal(torch.stack(old_states), torch.stack(states)))
                    self.assertEqual(old_actions, actions)
                    self.assertEqual(old_rewards, rewards)
                # The next batch must use fresh weights, rather than a stale worker policy.
                with torch.no_grad():
                    policy.layers[-2].bias[2] += 2.0
        finally:
            serial.close()
            parallel.close()


if __name__ == "__main__":
    unittest.main()
