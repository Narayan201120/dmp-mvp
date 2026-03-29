from __future__ import annotations

from dataclasses import dataclass, field

from daemon.adaptation_loop import AdaptationLoop
from daemon.eval_loop import EvalLoop
from daemon.messaging import MessageBus
from daemon.state import NodeState
from daemon.training_loop import TrainingLoop
from training.shard import TransformerShard


@dataclass(slots=True)
class Node:
    state: NodeState
    bus: MessageBus
    shard: TransformerShard | None = None
    training_loop: TrainingLoop = field(init=False)
    eval_loop: EvalLoop = field(init=False)
    adaptation_loop: AdaptationLoop = field(init=False)

    def __post_init__(self) -> None:
        self.training_loop = TrainingLoop(bus=self.bus, state=self.state, shard=self.shard)
        self.eval_loop = EvalLoop(bus=self.bus, state=self.state)
        self.adaptation_loop = AdaptationLoop()

    def start_loops(self) -> dict[str, object]:
        return {
            "training": self.training_loop,
            "eval": self.eval_loop,
            "adaptation": self.adaptation_loop,
        }

    def handle_lifecycle_events(self, *, version: int | None = None) -> None:
        if version is not None:
            self.state.version = version
