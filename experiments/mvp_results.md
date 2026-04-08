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

## 2026-04-03 Combined Precision Comparator

- Compression-only control:
- Reference artifact: `experiments/distributed_compression_top50_4bit_summary.json`
- Final eval loss: `2.139144106344743`
- Final eval loss delta vs baseline: `0.07579514113339503`

- `1 ms` latency with decay plus top-50% / 4-bit compression:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --max-staleness 2 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 4 --output-curve experiments\distributed_combined_top50_4bit_decay_1ms_curve.csv --output-summary experiments\distributed_combined_top50_4bit_decay_1ms_summary.json`
- Final eval loss: `2.2083787484602495`
- Final eval loss delta vs baseline: `0.14502978324890137`
- Incremental delta vs compression-only control: `0.06923464211550634`
- Boundary delivery: `200 delivered`, staleness multiplier min/max/avg `0.6065306597126334 / 0.6065306597126334 / 0.6065306597126334`, values kept min/max/avg `8192 / 8192 / 8192.0`

- `2 ms` latency with decay plus top-50% / 4-bit compression:
- Command: `python experiments\distributed_train.py --base-latency-ms 2 --max-staleness 4 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 4 --output-curve experiments\distributed_combined_top50_4bit_decay_2ms_curve.csv --output-summary experiments\distributed_combined_top50_4bit_decay_2ms_summary.json`
- Final eval loss: `2.484022682363337`
- Final eval loss delta vs baseline: `0.42067371715198876`
- Incremental delta vs compression-only control: `0.34487857601859373`
- Boundary delivery: `200 delivered`, staleness multiplier min/max/avg `0.36787944117144233 / 0.36787944117144233 / 0.36787944117144233`, values kept min/max/avg `8192 / 8192 / 8192.0`

- Jitter and reordering with decay plus top-50% / 4-bit compression:
- Command: `python experiments\distributed_train.py --base-latency-ms 1 --jitter-ms 1 --reorder-chance 0.5 --max-staleness 5 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 4 --output-curve experiments\distributed_combined_top50_4bit_decay_jitter_reorder_curve.csv --output-summary experiments\distributed_combined_top50_4bit_decay_jitter_reorder_summary.json`
- Final eval loss: `2.3596200942993164`
- Final eval loss delta vs baseline: `0.29627112908796827`
- Incremental delta vs compression-only control: `0.22047598795457324`
- Boundary delivery: `200 delivered`, delay min/max/avg `0 / 5 / 2.5 ms`, staleness multiplier min/max/avg `0.25 / 1.0 / 0.45084887223415804`, values kept min/max/avg `8192 / 8192 / 8192.0`

- Summary:
- `top50/4-bit` is slightly worse than `top50/6-bit` for the `1 ms` and jitter/reorder cases, but the gap remains small.
- Under the harsher `2 ms` setting, the `4-bit` and `6-bit` combined runs are effectively tied, which suggests staleness is dominating the outcome more than quantization precision there.
- That qualitative comparison is now backed by exact sparse payload byte reporting in the rerun artifacts below.

## 2026-04-03 Exact Sparse Wire-Byte Comparison

- Dense boundary tensor body:
- Reference size: `65536` bytes per delivered boundary

- Top-50% / 6-bit payload body:
- Reference artifacts: `experiments/distributed_compression_top50_6bit_summary.json`, `experiments/distributed_combined_top50_6bit_decay_1ms_summary.json`, `experiments/distributed_combined_top50_6bit_decay_2ms_summary.json`, `experiments/distributed_combined_top50_6bit_decay_jitter_reorder_summary.json`
- Sparse payload size: `22532` bytes per delivered boundary
- Wire ratio vs dense: `0.34381103515625`
- Final eval loss deltas vs baseline:
- zero delay: `0.06150285764174024`
- `1 ms` decay: `0.14099758321588673`
- `2 ms` decay: `0.42207824100147606`
- jitter/reorder decay: `0.2695574977181172`

- Top-50% / 4-bit payload body:
- Reference artifacts: `experiments/distributed_compression_top50_4bit_summary.json`, `experiments/distributed_combined_top50_4bit_decay_1ms_summary.json`, `experiments/distributed_combined_top50_4bit_decay_2ms_summary.json`, `experiments/distributed_combined_top50_4bit_decay_jitter_reorder_summary.json`
- Sparse payload size: `20484` bytes per delivered boundary
- Wire ratio vs dense: `0.31256103515625`
- Final eval loss deltas vs baseline:
- zero delay: `0.07579514113339503`
- `1 ms` decay: `0.14502978324890137`
- `2 ms` decay: `0.42067371715198876`
- jitter/reorder decay: `0.29627112908796827`

- Comparison:
- `top50/4-bit` saves `2048` bytes per delivered boundary relative to `top50/6-bit`, which is about `9.1%` smaller.
- Across the full `100`-step run with `200` delivered boundaries, that is `409600` fewer payload bytes.
- The byte savings come with a small quality cost at zero delay (`+0.01429`) and `1 ms` delay (`+0.00403`), are effectively neutral at `2 ms`, and are somewhat worse under jitter/reorder (`+0.02671`).

- Recommendation:
- If the goal is quality per byte, `top50/4-bit` is the better operating point.
- If the goal is the smallest quality penalty and transport cost is secondary, `top50/6-bit` remains the slightly safer point.

## 2026-04-08 Multi-Process Repeated-Step Smoke

- Command: `python experiments\process_window.py`
- Summary artifact: `experiments/process_window_summary.json`
- Runtime shape: `3` separate localhost worker processes, one shard per process, boundary handoff over TCP
- Input shape: `[2, 8]`
- Logits shape: `[2, 8, 32]`
- Max absolute difference vs reference model: `0.0`
- Forward result: `matches_reference = true`
- Repeated train-step result:
- versions exercised: `0, 1, 2`
- final process loss before: `2.6024177074432373`
- final process loss after: `2.370682954788208`
- final reference loss before: `2.6024177074432373`
- final reference loss after: `2.370682954788208`
- final train-step deltas: `0.0` before, `0.0` after
- full repeated-step result: `matches_reference = true`
- Worker state after the run:
- all workers at version `3`
- all workers with `last_checkpoint_version = 3`
- all workers retaining checkpoint versions `[0, 1, 2, 3]`

- Summary:
- This is the first hardware-facing slice beyond the in-process simulator: shard execution, reverse-order gradient flow, repeated updates, and checkpoint/version advancement all survive crossing real OS process boundaries and a simple socket protocol.
- The process runner now also exposes the same delay/staleness/compression knobs on the real worker handoff path, with coverage in the process-runtime tests.
- The process runner now also restores all workers to the last consistent checkpoint after a synthetic partial-commit failure, with the retry path matching the reference trainer exactly.
- The next gap is not basic cross-process training correctness anymore. It is using this rollback-capable process runtime for the first process-side comparison sweeps at the chosen operating point.

## 2026-04-08 First Process-Side Comparison Sweep

- Chosen operating point: `top50/4-bit`
- Rationale: still the best quality-per-byte point from the earlier simulator sweep, so it is the right first transport-efficient setting to exercise on the real worker handoff path.

- Compression-only process control:
- Command: `python experiments\process_window.py --compression-topk-ratio 0.5 --compression-num-bits 4 --output-summary experiments\process_window_top50_4bit_summary.json`
- Forward max-abs diff vs reference: `1.1463724374771118`
- Final process loss after step `2`: `2.4611024856567383`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.09041953086853027`
- Sparse boundary body: `424` bytes vs dense `1344` bytes, wire ratio `0.31547619047619047`

- `1 ms` decay plus `top50/4-bit`:
- Command: `python experiments\process_window.py --base-latency-ms 1 --max-staleness 2 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 4 --output-summary experiments\process_window_top50_4bit_decay_1ms_summary.json`
- Final process loss after step `2`: `2.5178654193878174`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.14718246459960938`
- Boundary delivery: `1 ms`, staleness multiplier `0.6065306597126334`, same `424`-byte sparse body

- `2 ms` decay plus `top50/4-bit`:
- Command: `python experiments\process_window.py --base-latency-ms 2 --max-staleness 4 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 4 --output-summary experiments\process_window_top50_4bit_decay_2ms_summary.json`
- Final process loss after step `2`: `2.788839340209961`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.41815638542175293`
- Boundary delivery: `2 ms`, staleness multiplier `0.36787944117144233`, same `424`-byte sparse body

- Summary:
- The process-side runtime now shows the same qualitative ordering as the simulator: compression alone causes a modest quality drop, `1 ms` decay is worse, and `2 ms` decay is substantially worse.
- That makes the rollback-capable process runtime good enough for the first real transport-facing comparisons, instead of treating the simulator as the only place where perturbations can be studied.

## 2026-04-08 Process-Side Precision Comparator

- Top-50% / 6-bit process control:
- Command: `python experiments\process_window.py --compression-topk-ratio 0.5 --compression-num-bits 6 --output-summary experiments\process_window_top50_6bit_summary.json`
- Final process loss after step `2`: `2.4419445991516113`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.07126164436340332`
- Sparse boundary body: `466` bytes vs dense `1344` bytes, wire ratio `0.34672619047619047`

- `1 ms` decay plus `top50/6-bit`:
- Command: `python experiments\process_window.py --base-latency-ms 1 --max-staleness 2 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 6 --output-summary experiments\process_window_top50_6bit_decay_1ms_summary.json`
- Final process loss after step `2`: `2.547651767730713`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.17696881294250488`

- `2 ms` decay plus `top50/6-bit`:
- Command: `python experiments\process_window.py --base-latency-ms 2 --max-staleness 4 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 6 --output-summary experiments\process_window_top50_6bit_decay_2ms_summary.json`
- Final process loss after step `2`: `2.7900149822235107`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.41933202743530273`

- Jitter/reorder decay plus `top50/6-bit`:
- Command: `python experiments\process_window.py --base-latency-ms 1 --jitter-ms 1 --reorder-chance 0.5 --max-staleness 5 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 6 --output-summary experiments\process_window_top50_6bit_decay_jitter_reorder_summary.json`
- Final process loss after step `2`: `2.747637987136841`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.3769550323486328`

- Jitter/reorder decay plus `top50/4-bit`:
- Command: `python experiments\process_window.py --base-latency-ms 1 --jitter-ms 1 --reorder-chance 0.5 --max-staleness 5 --staleness-decay-rate 0.5 --staleness-floor 0.25 --compression-topk-ratio 0.5 --compression-num-bits 4 --output-summary experiments\process_window_top50_4bit_decay_jitter_reorder_summary.json`
- Final process loss after step `2`: `2.85243558883667`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.4817526340484619`

- Comparison:
- `top50/4-bit` still has the smaller payload body at `424` bytes versus `466` bytes for `top50/6-bit`.
- On the process-side runtime, `top50/6-bit` is better at the compression-only control and much safer under jitter/reordering.
- `top50/4-bit` is slightly better at the fixed `1 ms` point and effectively tied at `2 ms`, so the same high-level tradeoff still holds: `4-bit` for smaller payloads, `6-bit` for a safer quality margin when transport gets messier.

- Rollback coverage:
- The process-runtime tests now also cover synthetic partial-commit failures at both `version=0` and `version=1`.
- In both cases, the runtime restores every worker to the last consistent checkpoint and the retried train step matches the in-process reference exactly.
- The process runtime now also survives a real worker exit between committed steps by relaunching the dead shard from the last committed checkpoint and matching the reference trainer on the retried step.

## 2026-04-08 External Worker Attach Smoke

- Hardware-facing default:
- Chosen operating point: `top50/6-bit`
- Rationale: `top50/4-bit` remains the better quality-per-byte point, but the external-worker path should bias toward the safer jitter/reorder margin before chasing the last `42` payload bytes.

- External-worker localhost control:
- Command shape: `python experiments\process_window.py --worker-endpoints <host:port,...> --stop-workers-on-close --compression-topk-ratio 0.5 --compression-num-bits 6 --output-summary experiments\process_window_external_top50_6bit_summary.json`
- Summary artifact: `experiments/process_window_external_top50_6bit_summary.json`
- Worker mode: `external`
- Final process loss after step `2`: `2.4419445991516113`
- Final reference loss after step `2`: `2.370682954788208`
- Final loss delta: `0.07126164436340332`
- Sparse boundary body: `466` bytes vs dense `1344` bytes, wire ratio `0.34672619047619047`
- Worker state after the run:
- all workers externally hosted on `127.0.0.1`
- all workers at version `3`
- all workers retaining checkpoint versions `[0, 1, 2, 3]`

- Summary:
- This is the first run where the parent attaches to already-running shard servers instead of spawning and owning the workers directly.
- That closes the main architectural gap between the localhost subprocess prototype and a real LAN deployment: the transport and training protocol no longer assume the coordinator and workers share one machine.
