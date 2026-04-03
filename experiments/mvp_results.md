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

## 2026-04-03 Delayed-Network Sweep

- `1 ms` fixed latency:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --max-staleness 2 --output-curve experiments\distributed_delay_1ms_curve.csv --output-summary experiments\distributed_delay_1ms_summary.json`
- Final eval loss: `2.063348965211348`
- Final eval loss delta vs baseline: `0.0`
- Boundary delivery: `200 delivered`, delay min/max/avg `1 / 1 / 1.0 ms`

- `2 ms` fixed latency:
- Command: `python experiments\distributed_train.py --base-latency-ms 2 --max-staleness 4 --output-curve experiments\distributed_delay_2ms_curve.csv --output-summary experiments\distributed_delay_2ms_summary.json`
- Final eval loss: `2.063348965211348`
- Final eval loss delta vs baseline: `0.0`
- Boundary delivery: `200 delivered`, delay min/max/avg `2 / 2 / 2.0 ms`

- `1 ms` latency with jitter and reordering:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --jitter-ms 1 --reorder-chance 0.5 --max-staleness 5 --output-curve experiments\distributed_delay_jitter_reorder_curve.csv --output-summary experiments\distributed_delay_jitter_reorder_summary.json`
- Final eval loss: `2.063348965211348`
- Final eval loss delta vs baseline: `0.0`
- Boundary delivery: `200 delivered`, delay min/max/avg `0 / 5 / 2.5 ms`

- Stale-drop edge check:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --max-staleness 0`
- Result: fails immediately with `StaleBoundaryError` before a comparison artifact is produced

- Summary:
- Within the current Phase 1 implementation, delivery timing changes the control path and fail-fast envelope but does not change the numerical training trajectory while messages remain within the configured staleness budget.

## 2026-04-03 Staleness-Weighted Sweep

- `1 ms` latency with decay:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --max-staleness 2 --staleness-decay-rate 0.5 --staleness-floor 0.25 --output-curve experiments\distributed_decay_1ms_curve.csv --output-summary experiments\distributed_decay_1ms_summary.json`
- Final eval loss: `2.160941568287936`
- Final eval loss delta vs baseline: `0.0975926030765879`
- Boundary delivery: `200 delivered`, staleness multiplier min/max/avg `0.6065306597126334 / 0.6065306597126334 / 0.6065306597126334`

- `2 ms` latency with decay:
- Command: `python experiments\distributed_train.py --base-latency-ms 2 --max-staleness 4 --staleness-decay-rate 0.5 --staleness-floor 0.25 --output-curve experiments\distributed_decay_2ms_curve.csv --output-summary experiments\distributed_decay_2ms_summary.json`
- Final eval loss: `2.4619074084542016`
- Final eval loss delta vs baseline: `0.3985584432428535`
- Boundary delivery: `200 delivered`, staleness multiplier min/max/avg `0.36787944117144233 / 0.36787944117144233 / 0.36787944117144233`

- Jitter and reordering with decay:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --jitter-ms 1 --reorder-chance 0.5 --max-staleness 5 --staleness-decay-rate 0.5 --staleness-floor 0.25 --output-curve experiments\distributed_decay_jitter_reorder_curve.csv --output-summary experiments\distributed_decay_jitter_reorder_summary.json`
- Final eval loss: `2.35578406940807`
- Final eval loss delta vs baseline: `0.2924351041967217`
- Boundary delivery: `200 delivered`, staleness multiplier min/max/avg `0.25 / 1.0 / 0.45084887223415804`

- Summary:
- Wiring the decay weight into the actual boundary signal makes in-budget latency numerically matter, and the degradation scales in the expected direction with stronger effective staleness.

## 2026-04-03 Compression Sweep

- Top-25% sparsification, 8-bit quantization, no error feedback:
- Command: `python experiments\distributed_train.py --compression-topk-ratio 0.25 --compression-num-bits 8 --output-curve experiments\distributed_compression_top25_8bit_curve.csv --output-summary experiments\distributed_compression_top25_8bit_summary.json`
- Final eval loss: `2.435922145843506`
- Final eval loss delta vs baseline: `0.3725731806321577`
- Boundary compression: `200` compressed deliveries, values kept min/max/avg `4096 / 4096 / 4096.0`

- Top-25% sparsification, 8-bit quantization, with error feedback:
- Command: `python experiments\distributed_train.py --compression-topk-ratio 0.25 --compression-num-bits 8 --compression-error-feedback --output-curve experiments\distributed_compression_top25_8bit_ef_curve.csv --output-summary experiments\distributed_compression_top25_8bit_ef_summary.json`
- Final eval loss: `3.0254006385803223`
- Final eval loss delta vs baseline: `0.9620516733689741`
- Boundary compression: `200` compressed deliveries, values kept min/max/avg `4096 / 4096 / 4096.0`

- Top-10% sparsification, 4-bit quantization, with error feedback:
- Command: `python experiments\distributed_train.py --compression-topk-ratio 0.1 --compression-num-bits 4 --compression-error-feedback --output-curve experiments\distributed_compression_top10_4bit_ef_curve.csv --output-summary experiments\distributed_compression_top10_4bit_ef_summary.json`
- Final eval loss: `3.1407880132848565`
- Final eval loss delta vs baseline: `1.0774390480735083`
- Boundary compression: `200` compressed deliveries, values kept min/max/avg `1639 / 1639 / 1639.0`

- Summary:
- Compression now perturbs the numerical training path end to end. On this toy setup, stronger compression degrades eval loss substantially, and the current error-feedback implementation is not helping yet.
- Follow-up:
- Boundary-activation error feedback is now treated as unsupported in the runtime because carrying residuals across independent batches distorts the signal. The EF artifacts above remain useful as diagnostic evidence of that failure mode.
