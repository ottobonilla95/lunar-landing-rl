# Experiment records

The published charts are preserved as supplied. All five have 5,000 episodes, with learning rates 0.001, 0.002, 0.003, 0.004, and 0.005 respectively. Other settings are unrecorded and stored as JSON null. There are no corresponding raw CSVs or checkpoints in this repository.

For each new experiment:

1. Choose a unique `--output models/<run>.pt` so earlier runs survive.
2. Save the exact command and a copy of `config.py` before training. CLI overrides are part of the effective settings.
3. Save console output, including initial/final greedy evaluation means.
4. Copy the chart and CSV into `results/<run>/` if you want them published. Keep large checkpoints local unless publication is explicitly requested.
5. Fill in the record below and add a row/image to the root README. Leave unavailable values unknown rather than inferring them from current defaults.

## New result template

- Run ID:
- Code commit:
- Exact command:
- Episodes completed:
- Hidden layers / activation:
- Batch size / workers:
- Learning rate / gamma / seed:
- Gradient clipping / PyTorch thread count:
- Python, dependency versions, OS, hardware, elapsed time:
- Initial/final greedy evaluation mean:
- Evaluation episode count / starting seed:
- Training chart / raw CSV:
- Config snapshot / console log:
- Notes and limitations:

Training reward uses sampled actions. The trainer prints separate greedy evaluation means before and after training on the same fixed starts. Repeatedly tuning against those starts makes them a development set; use additional fresh seeds for a final assessment. Single runs and peaks do not establish that one setup is best.
