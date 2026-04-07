from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
from typing import Any

import torch

from training.metrics import next_token_loss, split_next_token_batch
from training.shard import TransformerShard

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(len(payload).to_bytes(8, "big") + payload)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    payload_size = int.from_bytes(_recv_exact(sock, 8), "big")
    return pickle.loads(_recv_exact(sock, payload_size))


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise RuntimeError("socket connection closed before the full payload was received")
        chunks.extend(chunk)
    return bytes(chunks)


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True, slots=True)
class ProcessTrainWindowResult:
    version: int
    loss_before: float
    loss_after: float


@dataclass(slots=True)
class ProcessWindowRunner:
    shards: list[TransformerShard]
    learning_rate: float = 1e-2
    optimizer_name: str = "adam"
    weight_decay: float = 0.0
    startup_timeout_s: float = 10.0
    request_timeout_s: float = 10.0
    python_executable: str = sys.executable
    _ports: list[int] = field(init=False, repr=False, default_factory=list)
    _processes: list[subprocess.Popen[bytes]] = field(init=False, repr=False, default_factory=list)
    _started: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("at least one shard is required")
        if self.optimizer_name.lower() not in {"adam", "adamw"}:
            raise ValueError(f"unsupported optimizer_name: {self.optimizer_name}")

    def start(self) -> None:
        if self._started:
            return

        self._ports = [_reserve_port() for _ in self.shards]
        self._processes = []
        try:
            for shard, port in zip(self.shards, self._ports):
                process = subprocess.Popen(
                    [
                        self.python_executable,
                        "-m",
                        "daemon.socket_worker",
                        "--port",
                        str(port),
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                assert process.stdin is not None
                pickle.dump(
                    {
                        "shard": shard,
                        "learning_rate": self.learning_rate,
                        "optimizer_name": self.optimizer_name,
                        "weight_decay": self.weight_decay,
                    },
                    process.stdin,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                process.stdin.close()
                self._processes.append(process)

            self._wait_for_workers()
            self._started = True
        except Exception:
            self._terminate_processes()
            self._ports = []
            self._processes = []
            raise

    def close(self) -> None:
        if not self._started and not self._processes:
            return

        for port, process in zip(self._ports, self._processes):
            if process.poll() is not None:
                continue
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=self.request_timeout_s) as sock:
                    send_message(sock, {"kind": "stop"})
            except OSError:
                continue

        self._terminate_processes()

        self._ports = []
        self._processes = []
        self._started = False

    def run_window(self, input_ids: torch.Tensor, *, version: int) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if not self._started:
            self.start()

        message: dict[str, Any] = {
            "version": int(version),
            "tensor": input_ids.detach().to(device="cpu", dtype=torch.long).numpy(),
        }
        for index in range(len(self._ports)):
            message = self._request_worker(
                index,
                {
                    "kind": "run",
                    "version": int(message["version"]),
                    "tensor": message["tensor"],
                },
            )

        if message["kind"] != "logits":
            raise RuntimeError("final worker did not return logits")
        if int(message["version"]) != version:
            raise RuntimeError("worker pipeline returned a mismatched version")
        return torch.from_numpy(message["tensor"].copy())

    def evaluate_next_token_loss(self, input_ids: torch.Tensor, *, version: int) -> float:
        model_inputs, targets = split_next_token_batch(input_ids)
        with torch.no_grad():
            logits = self.run_window(model_inputs, version=version).to(dtype=torch.float32)
            loss = next_token_loss(logits, targets)
        return float(loss.item())

    def train_window(self, input_ids: torch.Tensor, *, version: int) -> ProcessTrainWindowResult:
        model_inputs, targets = split_next_token_batch(input_ids)
        loss_before = self.evaluate_next_token_loss(input_ids, version=version)

        message: dict[str, Any] = {
            "version": int(version),
            "tensor": model_inputs.detach().to(device="cpu", dtype=torch.long).numpy(),
        }
        for index in range(len(self._ports)):
            message = self._request_worker(
                index,
                {
                    "kind": "train_forward",
                    "version": int(message["version"]),
                    "tensor": message["tensor"],
                },
            )

        backward_message = self._request_worker(
            len(self._ports) - 1,
            {
                "kind": "backward_loss",
                "version": int(version),
                "target_ids": targets.detach().to(device="cpu", dtype=torch.long).numpy(),
            },
        )
        if backward_message["kind"] != "backward_grad":
            raise RuntimeError("last worker did not return a backward gradient message")

        for index in range(len(self._ports) - 2, -1, -1):
            backward_message = self._request_worker(
                index,
                {
                    "kind": "backward",
                    "version": int(version),
                    "tensor": backward_message["tensor"],
                },
            )
            if backward_message["kind"] != "backward_grad":
                raise RuntimeError(f"worker {index} did not return a backward gradient message")

        loss_after = self.evaluate_next_token_loss(input_ids, version=version + 1)
        return ProcessTrainWindowResult(version=version, loss_before=loss_before, loss_after=loss_after)

    def process_status(self) -> list[dict[str, object]]:
        if not self._started:
            return []
        return [
            {
                "pid": process.pid,
                "returncode": process.poll(),
                "alive": process.poll() is None,
            }
            for process in self._processes
        ]

    def _request_worker(self, index: int, request: dict[str, Any]) -> dict[str, Any]:
        port = self._ports[index]
        process = self._processes[index]
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=self.request_timeout_s) as sock:
                send_message(sock, request)
                return recv_message(sock)
        except (OSError, RuntimeError) as exc:
            if process.poll() is not None:
                raise RuntimeError(
                    f"worker for port {port} exited during request handling with code {process.returncode}: "
                    f"{self._read_process_stderr(process)}"
                ) from exc
            raise RuntimeError(f"worker for port {port} closed the request unexpectedly") from exc

    def _wait_for_workers(self) -> None:
        deadline = time.time() + self.startup_timeout_s
        pending = set(self._ports)
        while pending:
            for port, process in zip(self._ports, self._processes):
                if process.poll() is None:
                    continue
                raise RuntimeError(
                    f"worker for port {port} exited during startup with code {process.returncode}: "
                    f"{self._read_process_stderr(process)}"
                )

            if time.time() >= deadline:
                raise TimeoutError("timed out waiting for worker processes to open their ports")

            ready: set[int] = set()
            for port in pending:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2) as sock:
                        send_message(sock, {"kind": "ping"})
                        response = recv_message(sock)
                    if response.get("kind") == "ready":
                        ready.add(port)
                except (OSError, RuntimeError):
                    continue
            pending.difference_update(ready)
            if pending:
                time.sleep(0.05)

    def _terminate_processes(self) -> None:
        for process in self._processes:
            try:
                process.wait(timeout=self.request_timeout_s)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=self.request_timeout_s)

    def _read_process_stderr(self, process: subprocess.Popen[bytes]) -> str:
        if process.stderr is None:
            return "no stderr captured"
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return stderr or "no stderr output"

    def __enter__(self) -> ProcessWindowRunner:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
