from __future__ import annotations

import argparse
from dataclasses import dataclass
import pickle
import socket
import sys
from pathlib import Path

import torch
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daemon.process_runtime import recv_message, send_message
from training.metrics import next_token_loss


@dataclass(slots=True)
class ActiveWindow:
    input_hidden_states: torch.Tensor | None
    output_tensor: torch.Tensor


def _build_optimizer(
    shard: torch.nn.Module,
    *,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
) -> optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(shard.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(shard.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer_name: {optimizer_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a shard worker that serves forward requests over localhost TCP.")
    parser.add_argument("--port", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_config = pickle.load(sys.stdin.buffer)
    shard = launch_config["shard"]
    optimizer = _build_optimizer(
        shard,
        learning_rate=float(launch_config["learning_rate"]),
        optimizer_name=str(launch_config["optimizer_name"]),
        weight_decay=float(launch_config["weight_decay"]),
    )
    active_windows: dict[int, ActiveWindow] = {}
    shard.eval()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", args.port))
        server.listen()

        while True:
            connection, _ = server.accept()
            with connection:
                message = recv_message(connection)
                if message["kind"] == "ping":
                    send_message(connection, {"kind": "ready"})
                    continue
                if message["kind"] == "stop":
                    return
                if message["kind"] == "run":
                    shard.eval()
                    with torch.no_grad():
                        tensor = torch.from_numpy(message["tensor"].copy())
                        if shard.is_first_shard:
                            output = shard(input_ids=tensor.to(dtype=torch.long))
                        else:
                            output = shard(hidden_states=tensor.to(dtype=torch.float32))

                    send_message(
                        connection,
                        {
                            "kind": "logits" if shard.is_last_shard else "hidden_states",
                            "version": int(message["version"]),
                            "tensor": output.detach().to(device="cpu", dtype=torch.float32).numpy(),
                        },
                    )
                    continue

                if message["kind"] == "train_forward":
                    version = int(message["version"])
                    optimizer.zero_grad(set_to_none=True)
                    shard.train(True)

                    tensor = torch.from_numpy(message["tensor"].copy())
                    input_hidden_states: torch.Tensor | None = None
                    if shard.is_first_shard:
                        output = shard(input_ids=tensor.to(dtype=torch.long))
                    else:
                        input_hidden_states = tensor.to(dtype=torch.float32).requires_grad_(True)
                        output = shard(hidden_states=input_hidden_states)
                    active_windows[version] = ActiveWindow(
                        input_hidden_states=input_hidden_states,
                        output_tensor=output,
                    )

                    send_message(
                        connection,
                        {
                            "kind": "logits" if shard.is_last_shard else "hidden_states",
                            "version": version,
                            "tensor": output.detach().to(device="cpu", dtype=torch.float32).numpy(),
                        },
                    )
                    continue

                if message["kind"] == "backward_loss":
                    version = int(message["version"])
                    if not shard.is_last_shard:
                        raise RuntimeError("backward_loss is only valid on the last shard")
                    state = active_windows.pop(version)
                    targets = torch.from_numpy(message["target_ids"].copy()).to(dtype=torch.long)
                    loss = next_token_loss(state.output_tensor, targets)
                    loss.backward()
                    optimizer.step()
                    shard.eval()

                    input_grad = None
                    if state.input_hidden_states is not None:
                        assert state.input_hidden_states.grad is not None
                        input_grad = state.input_hidden_states.grad.detach().to(device="cpu", dtype=torch.float32).numpy()

                    send_message(
                        connection,
                        {
                            "kind": "backward_grad",
                            "version": version,
                            "loss": float(loss.item()),
                            "tensor": input_grad,
                        },
                    )
                    continue

                if message["kind"] == "backward":
                    version = int(message["version"])
                    state = active_windows.pop(version)
                    grad_output = torch.from_numpy(message["tensor"].copy()).to(dtype=torch.float32)
                    state.output_tensor.backward(grad_output)
                    optimizer.step()
                    shard.eval()

                    input_grad = None
                    if state.input_hidden_states is not None:
                        assert state.input_hidden_states.grad is not None
                        input_grad = state.input_hidden_states.grad.detach().to(device="cpu", dtype=torch.float32).numpy()

                    send_message(
                        connection,
                        {
                            "kind": "backward_grad",
                            "version": version,
                            "tensor": input_grad,
                        },
                    )
                    continue

                raise RuntimeError(f"unsupported worker message kind: {message['kind']}")


if __name__ == "__main__":
    main()
