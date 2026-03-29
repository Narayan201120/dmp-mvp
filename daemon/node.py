from __future__ import annotations

from dataclasses import dataclass, field

from torch import optim

from daemon.adaptation_loop import AdaptationLoop
from daemon.eval_loop import EvalLoop
from daemon.messaging import MessageBus
from daemon.state import NodeState
from daemon.training_loop import TrainingLoop
from training.checkpoints import SnapshotStore, module_state_to_numpy
from training.shard import TransformerShard


@dataclass(slots=True)
class Node:
    state: NodeState
    bus: MessageBus
    shard: TransformerShard | None = None
    learning_rate: float = 1e-2
    snapshot_depth: int = 10
    training_loop: TrainingLoop = field(init=False)
    eval_loop: EvalLoop = field(init=False)
    adaptation_loop: AdaptationLoop = field(init=False)
    optimizer: optim.Optimizer | None = field(init=False, default=None, repr=False)
    snapshot_store: SnapshotStore | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.training_loop = TrainingLoop(bus=self.bus, state=self.state, shard=self.shard)
        self.eval_loop = EvalLoop(bus=self.bus, state=self.state)
        self.adaptation_loop = AdaptationLoop()
        if self.shard is not None:
            self.optimizer = optim.Adam(self.shard.parameters(), lr=self.learning_rate)
            self.snapshot_store = SnapshotStore(max_depth=self.snapshot_depth)
            self.save_checkpoint(self.state.version)

    def start_loops(self) -> dict[str, object]:
        return {
            "training": self.training_loop,
            "eval": self.eval_loop,
            "adaptation": self.adaptation_loop,
        }

    def handle_lifecycle_events(self, *, version: int | None = None) -> None:
        if version is not None:
            self.state.version = version

    def set_training_mode(self, enabled: bool) -> None:
        if self.shard is None:
            raise RuntimeError("node has no shard attached")
        self.shard.train(enabled)

    def zero_grad(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("node optimizer is not initialized")
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("node optimizer is not initialized")
        self.optimizer.step()

    def save_checkpoint(self, version: int) -> None:
        if self.shard is None or self.snapshot_store is None:
            raise RuntimeError("node checkpointing requires an attached shard")
        self.snapshot_store.save(version, module_state_to_numpy(self.shard))
        self.state.last_checkpoint_version = version
