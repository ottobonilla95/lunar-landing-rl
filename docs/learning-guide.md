# How the network learns

For installation, commands, and recorded experiments, see the [project README](../README.md). Run the commands below from the repository root. Numerical architecture and tuning examples are examples; the current defaults live in `config.py`.

## The first neural network

```sh
uv run policy.py
```

This reads one initial state from Lunar Lander, feeds it through our network, prints the action probabilities, and samples one action. It doesn't open a window or train the network. To see the network act in the simulation window, use `uv run main.py --nn`.

Every meaningful line in `policy.py` has an explanation next to it. For example, `HIDDEN_SIZES = (64, 64)` and `ACTIVATION = "relu"` produce:

```text
8 state values → Linear(8, 64) → ReLU → Linear(64, 64) → ReLU
              → Linear(64, 4) → Softmax → 4 action probabilities
```

1. `torch` is PyTorch. A tensor is an array of numbers; PyTorch also computes the derivatives needed to train a network.
2. `nn.Module` is the base class for a network. `super().__init__()` lets PyTorch register its layers and parameters.
3. `nn.Sequential` applies layers in order. Each `nn.Linear` computes `output = weights × input + biases`. Training will change those weights and biases.
4. `nn.ReLU` returns `max(0, value)`. These nonlinear operations let the network represent more than a single linear transformation.
5. The final linear layer gives four unrestricted scores called logits. `nn.Softmax(dim=-1)` converts the last dimension (the actions) to positive probabilities that sum to one. A larger logit produces a larger probability.
6. `forward` describes the computation. Calling `policy(state)` runs it through PyTorch's module machinery.
7. `torch.manual_seed(config.SEED)` controls randomness for this example. It does not train the network. The initial weights are random.
8. `env.reset(seed=config.SEED)` gives us eight observations. `torch.tensor(..., dtype=torch.float32)` creates floating-point inputs compatible with the network's weights.
9. `torch.no_grad()` skips recording derivatives while we inspect the network. During training, the forward pass must track gradients.
10. `Categorical` represents the four action probabilities. `sample()` draws one action according to those probabilities; `.item()` turns the resulting scalar tensor into a Python integer.
11. The prints show the inputs and outputs; `finally` closes the environment. The example runs only when the file is executed directly, so we can later import `PolicyNetwork` into a training script.

Action order: `0` engines off, `1` left side engine, `2` main engine, `3` right side engine. Random initial weights give arbitrary probabilities, not useful landing decisions.

## Training the network

Edit **`config.py`** to change experiment settings in one place: hidden layer sizes, ReLU/Tanh activation, episodes, batch size, learning rate, gamma, seed, CPU threads, gradient clipping, evaluation starts/count, output paths, and chart settings. The environment's eight inputs and four actions are fixed interface values, also listed there. Then run:

```sh
uv run train.py
```

CLI options such as `--episodes` override the matching defaults for that run. Set `HIDDEN_SIZES = (32, 32)`, `(128, 128)`, or `(64,)` to try different networks, and `ACTIVATION = "tanh"` to try Tanh. Saved checkpoints record their architecture, so `main.py`, Netron export, and heatmaps load the saved model correctly even after the config changes. Original weights-only ReLU checkpoints still work.

The same commands now collect episodes in parallel, using `PARALLEL_WORKERS = 8` in `config.py`. The batch size is still the number of complete episodes per gradient update: with a batch of eight, eight persistent processes each collect one attempt using a copy of the current policy. The parent waits for all eight, computes returns separately for each attempt, and makes the same single update. Workers receive fresh weights at the start of every batch and never update parameters themselves. Sending complete episodes back avoids communication between processes on every rocket action.

Set `PARALLEL_WORKERS = 1` or pass `--workers 1` for sequential collection. The worker count is capped by the batch size and total episode count. Each episode has its own environment and action-sampling seed, and collected episodes are restored to their original order, so the new sequential and parallel modes are reproducible across worker counts with one PyTorch thread. Sampling differs from older versions, which shared one random sequence across all episodes. Starting worker processes adds a short one-time cost; long runs benefit more than tiny checks. Gradients, optimizer, episode counts, CSV, saved checkpoints, and automatic chart opening work as before.

On this laptop, collecting 128 episodes with the saved `32 → 32` policy and batch size eight took 5.54 seconds with one worker, 3.22 with two, 1.82 with four, and 1.11 with eight, after worker startup and one warm-up batch. All modes collected the same 124,658 steps and produced matching scores. This measures collection throughput, not total training time; learning updates, evaluation, plotting, and startup also take time.

For a short first run:

```sh
uv run train.py --episodes 32 --batch-size 8
```

For a longer experiment:

```sh
uv run train.py --episodes 1000 --batch-size 8 --output models/long_run.pt
uv run main.py --nn --model models/long_run.pt --greedy
```

Each invocation starts a fresh network. `--output` chooses the saved model file; rerunning with the same output replaces that file and its progress CSV. The CSV uses the same basename, such as `models/long_run.csv`. Ctrl+C during collection stops the run and saves the weights from the last completed update. The unfinished collection batch is discarded. We save network weights and architecture, not the optimizer state or an experience dataset.

Training also saves a plot with the same basename, such as `models/long_run.png`, using `plot_training.py` and Matplotlib. After final evaluation it automatically opens the chart in the computer's image viewer (normally Preview on macOS). Set `AUTO_OPEN_PLOT = False` in `config.py` or pass `--no-open-plot` to save without opening. The top panel shows loss; the bottom shows the mean training reward in each collection batch. Faint lines show individual updates; bold lines show a trailing average, controlled by `PLOT_SMOOTHING_WINDOW` (default ten updates). This REINFORCE loss can rise, fall, or become negative: a decreasing loss across different batches is not evidence that landings improve. Check reward trends and the separate initial/final evaluation scores.

To plot an existing run without retraining:

```sh
uv run plot_training.py models/long_run.csv
```

The learning rate and discount factor are now command-line options, which override `config.py` for that run:

```sh
uv run train.py --episodes 20000 --batch-size 16 --learning-rate 0.0003 --gamma 0.99 --output models/experiment.pt
```

Useful comparisons include learning rates `0.0003` versus `0.001`, batch sizes `8`, `16`, and `32`, and discount factors `0.98`, `0.99`, and `0.995`. These are experiments, not guaranteed improvements. Change one setting at a time and repeat with several seeds. Larger batches mean fewer updates for the same episode count.

For architecture experiments, try hidden layers `32 → 32` (smaller), `128 → 128` (wider), or one hidden layer of `64` (shallower). Change `HIDDEN_SIZES` in `config.py`; `policy.py` builds the matching layers. Training starts fresh with the selected architecture. Saved models retain their own original architecture when loaded. ReLU versus Tanh is another experiment. Architecture tuning is separate from changing the learning algorithm, such as adding a value network or using PPO.

We use **REINFORCE with reward-to-go**, a simple policy-gradient method, with returns centered and scaled across each batch. PPO was an earlier example of an algorithm we could use; this implementation keeps the learning math visible without adding a second value network. This method can be noisy and may need substantial training and tuning to land consistently. A short run checks that training works; it does not establish that the environment is solved.

The exact sequence is:

```text
Collect 8 complete attempts with unchanged weights (8 workers by default)
→ calculate future rewards within each attempt
→ update weights once
→ discard that batch and collect 8 new attempts
```

Training functions are explained inline in `train.py` and `rollouts.py`. Read the functions in this order:

### 1. `discounted_returns`: give earlier actions credit for later outcomes

An immediate engine penalty does not tell us whether firing the engine prevented a later crash. We therefore calculate a future-reward total for every action:

```text
return[t] = reward[t] + gamma × return[t + 1]
```

`gamma = 0.99` means a reward one step later counts 99% as much, two steps later 98.01%, and so on. For the tiny example `[1, 2, 3]` with `gamma = 0.5`, the results are `[2.75, 3.5, 3]`. The code starts at the end, accumulates backward, and reverses the result. It computes this separately for each attempt so rewards never leak into a previous attempt. In this introductory trainer, the 1,000-step time limit is treated as the end of a finite attempt, with zero return beyond it; there is no value network estimating what would happen afterward.

### 2. `collect_episode`: gather experience

`collect_episode` lives in `rollouts.py`. `reset` starts an attempt. Each loop turns the current observation into a tensor, gets action probabilities, samples an action, and calls `env.step`. The three lists keep the state before the action, the chosen action, and its resulting reward in matching order. Either end flag stops the attempt. The sampling seed is set per episode inside `torch.random.fork_rng`, which restores the caller's RNG afterward. `torch.no_grad()` is appropriate here because weights do not change during collection; the training function recomputes probabilities with gradients enabled. `EpisodeCollector` runs this function in reusable processes and returns complete episodes in order.

### 3. `update_policy`: convert experience into a weight update

`torch.stack(states)` creates a matrix with one row per step and eight columns. Action IDs become an integer tensor; future returns become a float tensor. We subtract the batch's mean return and divide by its population standard deviation plus `1e-8`. These training weights express above-average or below-average outcomes and keep the scale manageable. This is a simple batch comparison, not a learned state-value estimate. Identical returns give zero training weights.

We run all recorded states through the network again. `distribution.log_prob(actions)` retrieves the **logarithm of the probability of the action actually taken** in each state. It does not sample new actions. The loss is:

```python
loss = -(log_probabilities * weights).mean()
```

For a positive training weight, minimizing this expression encourages increasing that action's probability in its recorded state. A negative weight encourages decreasing it. The minus sign converts reward maximization into the loss minimization expected by the optimizer. Updates share network parameters, so individual probabilities can also be affected by other samples. One batch does not guarantee better landings.

`optimizer.zero_grad()` clears old derivatives. `loss.backward()` calculates new derivatives through our network. We do not differentiate through the physics or the random sampling. `clip_grad_norm_` limits the overall gradient size to 1. `optimizer.step()` is the line that actually changes the weights and biases; Adam manages the update using the configured learning rate. The loss number is a training calculation, not the landing score, and lower losses across different batches do not demonstrate better performance.

### 4. `evaluate`: check behavior without learning

Before and after training, this function runs `EVALUATION_EPISODES` attempts (default ten) starting at `EVALUATION_SEED`, using the same test seeds in a separate environment. It chooses the most likely action with `argmax` and sums the original rewards without discounting. Its mean score is an indicator of behavior on those starts; it is not a guarantee of general performance. Evaluation's deterministic actions differ from training's sampled actions. `main.py --greedy` uses the same action selection as evaluation. A mean around 200 or better is a useful target, but consistent results on many fresh seeds are needed to judge a trained policy.

### 5. `main`: organize the experiment

The command-line parser reads the attempt count, attempts per batch, random seed, and output path, then validates the inputs. We select one CPU thread for this small network, seed PyTorch, create the policy and Adam optimizer, and choose the discount factor. An initial evaluation gives us a comparison score.

The outer loop creates a fresh batch and passes that batch's episode seeds to `EpisodeCollector`. The collector runs the attempts without updating weights and returns them in order; `extend` combines their states, actions, and separately calculated future returns. A final batch may contain fewer attempts. One call to `update_policy` trains on that batch, after which we print its mean collection score and loss and write a CSV row. Collection scores describe behavior **before** that batch's update. The next batch uses the newly updated network. The workers are shut down when training finishes or is interrupted.

`save_policy(...)` uses `torch.save` to write the final weights, hidden layer sizes, and activation name. In `main.py`, `load_policy(...)` reads that checkpoint with `weights_only=True`, builds its architecture, and restores its parameters. Original weights-only checkpoints are also supported: their layer widths are inferred from tensor shapes, and their activation is ReLU. `policy.eval()` selects inference behavior; `torch.no_grad()` separately disables gradient recording. Neither performs training. The last evaluation repeats the configured test starts, then the saved loss/reward chart opens automatically.

The `if __name__ == "__main__"` block starts training only when you execute the file directly. Importing its functions for tests does not start a run.

The implementation follows the policy-gradient idea explained in [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html). The update mechanics are covered by [PyTorch's optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

## Checks

```sh
uv run python -m unittest discover -s tests -v
```

These check future-reward calculations, the direction of learning updates, stable handling of identical returns, checkpoint compatibility, configuration and chart opening, stopping collection on either end flag, and identical sequential/parallel experience including partial batches and updated policy weights.
