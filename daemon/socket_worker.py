from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import os
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
from training.checkpoints import SnapshotStore, module_state_to_numpy, numpy_state_to_module
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
    parser = argparse.ArgumentParser(description="Run a shard worker that serves forward requests over TCP.")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    return parser.parse_args()


def _read_launch_config_from_stdin() -> dict[str, object] | None:
    data = sys.stdin.buffer.read()
    if not data:
        return None
    return pickle.loads(data)


def _initialize_runtime_state(launch_config: dict[str, object]) -> dict[str, object]:
    shard = launch_config["shard"]
    snapshot_depth = int(launch_config["snapshot_depth"])
    optimizer = _build_optimizer(
        shard,
        learning_rate=float(launch_config["learning_rate"]),
        optimizer_name=str(launch_config["optimizer_name"]),
        weight_decay=float(launch_config["weight_decay"]),
    )
    snapshot_store = SnapshotStore(max_depth=snapshot_depth)
    optimizer_snapshots: OrderedDict[int, bytes] = OrderedDict()

    def save_optimizer_snapshot(version: int) -> None:
        optimizer_snapshots[version] = pickle.dumps(optimizer.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
        while len(optimizer_snapshots) > snapshot_depth:
            optimizer_snapshots.popitem(last=False)

    if "model_snapshots" in launch_config and "optimizer_snapshots" in launch_config:
        for version, state in launch_config["model_snapshots"]:
            snapshot_store.save(int(version), state)
        optimizer_snapshots.update(
            OrderedDict(
                (int(version), bytes(state))
                for version, state in launch_config["optimizer_snapshots"]
            )
        )
        current_version = int(launch_config["current_version"])
        numpy_state_to_module(shard, snapshot_store.restore(current_version))
        if current_version not in optimizer_snapshots:
            raise KeyError(f"optimizer snapshot for version {current_version} is unavailable during startup")
        optimizer.load_state_dict(pickle.loads(optimizer_snapshots[current_version]))
        last_checkpoint_version = int(launch_config.get("last_checkpoint_version", current_version))
    else:
        current_version = 0
        snapshot_store.save(current_version, module_state_to_numpy(shard))
        save_optimizer_snapshot(current_version)
        last_checkpoint_version = current_version
    shard.eval()
    return {
        "shard": shard,
        "snapshot_depth": snapshot_depth,
        "optimizer": optimizer,
        "snapshot_store": snapshot_store,
        "optimizer_snapshots": optimizer_snapshots,
        "save_optimizer_snapshot": save_optimizer_snapshot,
        "current_version": current_version,
        "last_checkpoint_version": last_checkpoint_version,
        "pending_update_version": None,
        "active_windows": {},
    }


def _export_status(runtime_state: dict[str, object], *, bind_host: str, port: int) -> dict[str, object]:
    snapshot_store = runtime_state["snapshot_store"]
    return {
        "kind": "status",
        "pid": os.getpid(),
        "host": bind_host,
        "port": port,
        "version": runtime_state["current_version"],
        "last_checkpoint_version": runtime_state["last_checkpoint_version"],
        "checkpoint_versions": snapshot_store.versions(),
    }


def _export_state(runtime_state: dict[str, object], *, bind_host: str, port: int) -> dict[str, object]:
    snapshot_store = runtime_state["snapshot_store"]
    optimizer_snapshots = runtime_state["optimizer_snapshots"]
    return {
        "kind": "export_state",
        "pid": os.getpid(),
        "host": bind_host,
        "port": port,
        "version": runtime_state["current_version"],
        "last_checkpoint_version": runtime_state["last_checkpoint_version"],
        "checkpoint_versions": snapshot_store.versions(),
        "model_snapshots": [
            (version, snapshot_store.restore(version))
            for version in snapshot_store.versions()
        ],
        "optimizer_snapshots": list(optimizer_snapshots.items()),
    }


def main() -> None:
    args = parse_args()
    launch_config = _read_launch_config_from_stdin()
    runtime_state = _initialize_runtime_state(launch_config) if launch_config is not None else None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind_host, args.port))
        server.listen()

        while True:
            connection, _ = server.accept()
            with connection:
                message = recv_message(connection)

                if message["kind"] == "ping":
                    send_message(connection, {"kind": "ready"})
                    continue

                if message["kind"] == "configure":
                    runtime_state = _initialize_runtime_state(message["config"])
                    send_message(
                        connection,
                        {
                            "kind": "configured",
                            "pid": os.getpid(),
                            "host": args.bind_host,
                            "port": args.port,
                            "version": runtime_state["current_version"],
                        },
                    )
                    continue

                if runtime_state is None:
                    raise RuntimeError("worker must be configured before handling runtime messages")

                shard = runtime_state["shard"]
                optimizer = runtime_state["optimizer"]
                snapshot_store = runtime_state["snapshot_store"]
                optimizer_snapshots = runtime_state["optimizer_snapshots"]
                save_optimizer_snapshot = runtime_state["save_optimizer_snapshot"]
                current_version = int(runtime_state["current_version"])
                last_checkpoint_version = int(runtime_state["last_checkpoint_version"])
                pending_update_version = runtime_state["pending_update_version"]
                active_windows: dict[int, ActiveWindow] = runtime_state["active_windows"]

                if message["kind"] == "status":
                    send_message(connection, _export_status(runtime_state, bind_host=args.bind_host, port=args.port))
                    continue

                if message["kind"] == "export_state":
                    send_message(connection, _export_state(runtime_state, bind_host=args.bind_host, port=args.port))
                    continue

                if message["kind"] == "stop":
                    return

                if message["kind"] == "run":
                    if int(message["version"]) != current_version:
                        raise RuntimeError(
                            f"worker version mismatch for run: got {message['version']}, expected {current_version}"
                        )
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
                    if version != current_version:
                        raise RuntimeError(
                            f"worker version mismatch for train_forward: got {version}, expected {current_version}"
                        )
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
                    runtime_state["pending_update_version"] = None

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

                    input_grad = None
                    if state.input_hidden_states is not None:
                        assert state.input_hidden_states.grad is not None
                        input_grad = state.input_hidden_states.grad.detach().to(device="cpu", dtype=torch.float32).numpy()
                    runtime_state["pending_update_version"] = version

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

                    input_grad = None
                    if state.input_hidden_states is not None:
                        assert state.input_hidden_states.grad is not None
                        input_grad = state.input_hidden_states.grad.detach().to(device="cpu", dtype=torch.float32).numpy()
                    runtime_state["pending_update_version"] = version

                    send_message(
                        connection,
                        {
                            "kind": "backward_grad",
                            "version": version,
                            "tensor": input_grad,
                        },
                    )
                    continue

                if message["kind"] == "commit":
                    version = int(message["version"])
                    if runtime_state["pending_update_version"] != version:
                        raise RuntimeError(
                            f"worker commit mismatch: got {version}, pending {runtime_state['pending_update_version']}"
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    shard.eval()
                    runtime_state["current_version"] = version + 1
                    snapshot_store.save(runtime_state["current_version"], module_state_to_numpy(shard))
                    save_optimizer_snapshot(runtime_state["current_version"])
                    runtime_state["last_checkpoint_version"] = runtime_state["current_version"]
                    runtime_state["pending_update_version"] = None

                    send_message(
                        connection,
                        {
                            "kind": "committed",
                            "version": runtime_state["current_version"],
                            "last_checkpoint_version": runtime_state["last_checkpoint_version"],
                            "checkpoint_versions": snapshot_store.versions(),
                        },
                    )
                    continue

                if message["kind"] == "rollback":
                    version = int(message["version"])
                    restored_state = snapshot_store.restore(version)
                    numpy_state_to_module(shard, restored_state)
                    snapshot_store.discard_after(version)
                    for snapshot_version in list(optimizer_snapshots.keys()):
                        if snapshot_version > version:
                            del optimizer_snapshots[snapshot_version]
                    if version not in optimizer_snapshots:
                        raise KeyError(f"optimizer snapshot for version {version} is unavailable")
                    optimizer.load_state_dict(pickle.loads(optimizer_snapshots[version]))
                    optimizer.zero_grad(set_to_none=True)
                    shard.eval()
                    runtime_state["current_version"] = version
                    runtime_state["last_checkpoint_version"] = version
                    runtime_state["pending_update_version"] = None
                    active_windows.clear()
                    send_message(
                        connection,
                        {
                            "kind": "rolled_back",
                            "version": runtime_state["current_version"],
                            "last_checkpoint_version": runtime_state["last_checkpoint_version"],
                            "checkpoint_versions": snapshot_store.versions(),
                        },
                    )
                    continue

                if message["kind"] == "abort":
                    version = int(message["version"])
                    active_windows.pop(version, None)
                    optimizer.zero_grad(set_to_none=True)
                    shard.eval()
                    if runtime_state["pending_update_version"] == version:
                        runtime_state["pending_update_version"] = None
                    send_message(connection, {"kind": "aborted", "version": runtime_state["current_version"]})
                    continue

                raise RuntimeError(f"unsupported worker message kind: {message['kind']}")


if __name__ == "__main__":
    main()
