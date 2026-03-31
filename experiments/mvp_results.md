# MVP Results

## 2026-03-30 Zero-Delay Distributed Comparison

- Command: `python experiments\distributed_train.py`
- Distributed curve: `experiments/distributed_loss_curve.csv`
- Distributed summary: `experiments/distributed_run_summary.json`
- Baseline reference: `baseline/baseline_loss_curve.csv` and `baseline/baseline_config.json`
- Initial eval loss: `3.8332548574967817`
- Final distributed eval loss: `2.063348965211348`
- Final baseline eval loss: `2.063348965211348`
- Final eval loss delta: `0.0`
- Eval loss ratio: `1.0`
- Note: the train-loss checkpoints differ from the centralized curve only by tiny floating-point noise at a few steps, while the eval curve matches exactly.
