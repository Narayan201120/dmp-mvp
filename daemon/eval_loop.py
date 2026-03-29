from __future__ import annotations

from dataclasses import dataclass

from daemon.messaging import MessageBus
from daemon.state import NodeState


@dataclass(slots=True)
class EvalLoop:
    bus: MessageBus
    state: NodeState

    async def run_canonical_eval(self, loss: float) -> float:
        self.state.latest_eval_loss = loss
        await self.bus.publish(
            "eval.metrics",
            {"node_id": self.state.node_id, "version": self.state.version, "loss": loss},
        )
        return loss

