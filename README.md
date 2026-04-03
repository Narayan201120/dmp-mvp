# dmp-mvp

Phase 1 MVP for a decentralized, layer-sharded transformer training system.

## What This Repo Is

This project is a localhost-first prototype of the DMP design:

- shard a transformer across multiple nodes
- exchange versioned boundary state between adjacent shards
- study staleness, compression, snapshots, and rollback
- compare the distributed run against a centralized baseline

The current implementation is intentionally narrower than the full design document. Real networking, trust systems, proof-of-contribution, and volunteer-node deployment are out of scope for the first milestone.

## Current Status

- repo scaffolded
- toy transformer implemented
- shard equivalence proven
- single-window runtime path verified
- test suite passing (`26 passed`)
- eval artifact locked
- centralized baseline runner implemented
- first frozen baseline run completed
- distributed comparison runner implemented
- first zero-delay distributed comparison run completed
- zero-delay distributed final eval loss matches the frozen baseline (`2.063348965211348`)
- delayed-network comparison sweeps completed for latency, jitter, and reordering within the staleness budget
- all successful delayed sweeps remain eval-curve identical to the zero-delay control and frozen baseline
- the `1 ms` / `max_staleness=0` edge still fails fast with a stale-boundary error as designed
- staleness-weighted delayed sweeps now produce measurable eval-loss deltas against baseline
- end-to-end compression sweeps now produce measurable eval-loss deltas against baseline and record boundary density stats
- boundary-activation error feedback is now rejected explicitly; the older EF artifacts are retained only as diagnostic evidence
- supported non-EF compression tuning shows that keeping more boundary values matters more than raising quantization precision on this toy setup
- first combined staleness-plus-compression matrix is complete for the supported `top50/6-bit` point
- `top50/4-bit` versus `top50/6-bit` combined-delay comparison is complete, and the gap stays small once staleness becomes the dominant source of damage
- exact sparse payload byte reporting is now wired into the experiment summaries
- current byte-aware recommendation: `top50/4-bit` for better quality per byte, `top50/6-bit` for a slightly safer quality margin under jitter

## Current Test Command

```powershell
python -m pytest tests -q -p no:cacheprovider
```

## Project Layout

```text
baseline/     centralized reference training scaffold
daemon/       runtime coordination and loop scaffolding
eval/         frozen eval artifact
experiments/  run outputs
sim/          network emulation helpers
tests/        protocol and correctness tests
training/     core math, protocol, shard, and state logic
```

## Key Artifacts

- `DMP_Final_Project_Document.docx`
- `PHASE1_EXECUTION_PLAN.md`
- `PROGRESS.md`

## Immediate Next Step

Use the current byte-aware tradeoff artifacts to pick the operating point for future sweeps, then retune staleness around that choice:

- centralized reference artifacts:
  - `baseline/baseline_loss_curve.csv`
  - `baseline/baseline_config.json`
- zero-delay distributed control artifacts:
  - `experiments/distributed_loss_curve.csv`
  - `experiments/distributed_run_summary.json`
- delayed comparison artifacts:
  - `experiments/distributed_delay_1ms_summary.json`
  - `experiments/distributed_delay_2ms_summary.json`
  - `experiments/distributed_delay_jitter_reorder_summary.json`
- staleness-weighted artifacts:
  - `experiments/distributed_decay_1ms_summary.json`
  - `experiments/distributed_decay_2ms_summary.json`
  - `experiments/distributed_decay_jitter_reorder_summary.json`
- compression artifacts:
  - `experiments/distributed_compression_top25_8bit_summary.json`
  - `experiments/distributed_compression_top50_8bit_summary.json`
  - `experiments/distributed_compression_top50_6bit_summary.json`
  - `experiments/distributed_compression_top50_4bit_summary.json`
  - `experiments/distributed_compression_top25_12bit_summary.json`
- combined tradeoff artifacts:
  - `experiments/distributed_combined_top50_6bit_decay_1ms_summary.json`
  - `experiments/distributed_combined_top50_6bit_decay_2ms_summary.json`
  - `experiments/distributed_combined_top50_6bit_decay_jitter_reorder_summary.json`
  - `experiments/distributed_combined_top50_4bit_decay_1ms_summary.json`
  - `experiments/distributed_combined_top50_4bit_decay_2ms_summary.json`
  - `experiments/distributed_combined_top50_4bit_decay_jitter_reorder_summary.json`
- diagnostic-only unsupported EF artifacts:
  - `experiments/distributed_compression_top25_8bit_ef_summary.json`
  - `experiments/distributed_compression_top10_4bit_ef_summary.json`
