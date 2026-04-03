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

## 2026-04-03 Supported Compression Tuning

- Top-50% sparsification, 8-bit quantization:
- Command: `python experiments\distributed_train.py --compression-topk-ratio 0.5 --compression-num-bits 8 --output-curve experiments\distributed_compression_top50_8bit_curve.csv --output-summary experiments\distributed_compression_top50_8bit_summary.json`
- Final eval loss: `2.1229094700379805`
- Final eval loss delta vs baseline: `0.05956050482663233`
- Boundary compression: `200` compressed deliveries, values kept min/max/avg `8192 / 8192 / 8192.0`

- Top-25% sparsification, 12-bit quantization:
- Command: `python experiments\distributed_train.py --compression-topk-ratio 0.25 --compression-num-bits 12 --output-curve experiments\distributed_compression_top25_12bit_curve.csv --output-summary experiments\distributed_compression_top25_12bit_summary.json`
- Final eval loss: `2.490146051753651`
- Final eval loss delta vs baseline: `0.42679708654230275`
- Boundary compression: `200` compressed deliveries, values kept min/max/avg `4096 / 4096 / 4096.0`

- Top-50% sparsification, 6-bit quantization:
- Command: `python experiments\distributed_train.py --compression-topk-ratio 0.5 --compression-num-bits 6 --output-curve experiments\distributed_compression_top50_6bit_curve.csv --output-summary experiments\distributed_compression_top50_6bit_summary.json`
- Final eval loss: `2.1248518228530884`
- Final eval loss delta vs baseline: `0.06150285764174024`
- Boundary compression: `200` compressed deliveries, values kept min/max/avg `8192 / 8192 / 8192.0`

- Top-50% sparsification, 4-bit quantization:
- Command: `python experiments\distributed_train.py --compression-topk-ratio 0.5 --compression-num-bits 4 --output-curve experiments\distributed_compression_top50_4bit_curve.csv --output-summary experiments\distributed_compression_top50_4bit_summary.json`
- Final eval loss: `2.139144106344743`
- Final eval loss delta vs baseline: `0.07579514113339503`
- Boundary compression: `200` compressed deliveries, values kept min/max/avg `8192 / 8192 / 8192.0`

- Summary:
- On this toy setup, preserving more boundary values matters more than increasing quantization precision. The denser `top50/*` runs beat the sparser `top25/*` runs even when the lower-bit runs use fewer quantization bits.
- `top50/6-bit` is nearly identical to `top50/8-bit`, and `top50/4-bit` still clearly outperforms the old `top25/8-bit` point.
- The current artifacts track kept values, not full sparse wire bytes, so this is a quality-versus-compression proxy rather than a full transport-cost measurement.

## 2026-04-03 Combined Staleness Plus Compression Sweep

- Compression-only control:
- Reference artifact: `experiments/distributed_compression_top50_6bit_summary.json`
- Final eval loss: `2.1248518228530884`
- Final eval loss delta vs baseline: `0.06150285764174024`

- `1 ms` latency with decay plus top-50% / 6-bit compression:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --max-staleness 2 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 6 --output-curve experiments\distributed_combined_top50_6bit_decay_1ms_curve.csv --output-summary experiments\distributed_combined_top50_6bit_decay_1ms_summary.json`
- Final eval loss: `2.204346548427235`
- Final eval loss delta vs baseline: `0.14099758321588673`
- Incremental delta vs compression-only control: `0.07949472557414646`
- Boundary delivery: `200 delivered`, staleness multiplier min/max/avg `0.6065306597126334 / 0.6065306597126334 / 0.6065306597126334`, values kept min/max/avg `8192 / 8192 / 8192.0`

- `2 ms` latency with decay plus top-50% / 6-bit compression:
- Command: `python experiments\distributed_train.py --base-latency-ms 2 --max-staleness 4 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 6 --output-curve experiments\distributed_combined_top50_6bit_decay_2ms_curve.csv --output-summary experiments\distributed_combined_top50_6bit_decay_2ms_summary.json`
- Final eval loss: `2.485427206212824`
- Final eval loss delta vs baseline: `0.42207824100147606`
- Incremental delta vs compression-only control: `0.36057538335973546`
- Boundary delivery: `200 delivered`, staleness multiplier min/max/avg `0.36787944117144233 / 0.36787944117144233 / 0.36787944117144233`, values kept min/max/avg `8192 / 8192 / 8192.0`

- Jitter and reordering with decay plus top-50% / 6-bit compression:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --jitter-ms 1 --reorder-chance 0.5 --max-staleness 5 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 6 --output-curve experiments\distributed_combined_top50_6bit_decay_jitter_reorder_curve.csv --output-summary experiments\distributed_combined_top50_6bit_decay_jitter_reorder_summary.json`
- Final eval loss: `2.3329064629294654`
- Final eval loss delta vs baseline: `0.2695574977181172`
- Incremental delta vs compression-only control: `0.20805464007637705`
- Boundary delivery: `200 delivered`, delay min/max/avg `0 / 5 / 2.5 ms`, staleness multiplier min/max/avg `0.25 / 1.0 / 0.45084887223415804`, values kept min/max/avg `8192 / 8192 / 8192.0`

- Summary:
- The effects compound in the expected order: `1 ms` is mildest, jitter/reorder is intermediate, and `2 ms` is the worst combined setting.
- The supported `top50/6-bit` point remains a good anchor for combined sweeps because its zero-delay cost is small, which makes the added staleness penalty easy to interpret.
- The summaries still report kept values rather than exact sparse wire bytes, so the current matrix is quality-first rather than a full transport-efficiency comparison.
