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
- test suite passing
- eval artifact locked
- centralized baseline runner implemented
- first frozen baseline run completed

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

Use the frozen centralized baseline as the reference for distributed comparisons:

- baseline step budget: `100`
- eval cadence: every `10` steps
- reference artifacts:
  - `baseline/baseline_loss_curve.csv`
  - `baseline/baseline_config.json`
