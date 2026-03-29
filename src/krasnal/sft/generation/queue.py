from __future__ import annotations

import multiprocessing as mp
import queue
import signal
from pathlib import Path

from krasnal.sft.generation.source import OnlineCotDataSource, load_raw_games


def derive_worker_seed(seed: int, worker_index: int) -> int:
    """Derive a deterministic per-worker seed."""
    return seed + (worker_index + 1) * 10_000


def configure_worker_signals() -> None:
    """Leave Ctrl-C handling to the parent trainer process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _producer_loop(
    *,
    worker_index: int,
    seed: int,
    stockfish_path: str,
    multipv_min: int,
    multipv_max: int,
    depth: int,
    max_len: int,
    row_queue: mp.Queue,
    stop_event: mp.Event,
    accepted_value: mp.Value,
    queue_depth_value: mp.Value,
) -> None:
    """Background worker loop that produces CoT rows."""
    configure_worker_signals()
    try:
        games = load_raw_games()
        with OnlineCotDataSource(
            games=games,
            stockfish_path=Path(stockfish_path),
            multipv_min=multipv_min,
            multipv_max=multipv_max,
            depth=depth,
            seed=derive_worker_seed(seed, worker_index),
            max_len=max_len,
        ) as source:
            while not stop_event.is_set():
                row = source.sample_row()
                while not stop_event.is_set():
                    try:
                        row_queue.put(row, timeout=0.2)
                    except queue.Full:
                        continue
                    with accepted_value.get_lock():
                        accepted_value.value += 1
                    with queue_depth_value.get_lock():
                        queue_depth_value.value += 1
                    break
    except KeyboardInterrupt:
        return


class CotProducerPool:
    """Manage background CoT row producers and a bounded row queue."""

    def __init__(
        self,
        *,
        num_producers: int,
        queue_max_rows: int,
        stockfish_path: Path,
        multipv_min: int,
        multipv_max: int,
        depth: int,
        max_len: int,
        seed: int,
    ) -> None:
        if num_producers <= 0:
            raise ValueError("num_producers must be > 0")
        self.ctx = mp.get_context("spawn")
        self.row_queue = self.ctx.Queue(maxsize=queue_max_rows)
        self.stop_event = self.ctx.Event()
        self.accepted_value = self.ctx.Value("q", 0)
        self.queue_depth_value = self.ctx.Value("q", 0)
        self.processes = [
            self.ctx.Process(
                target=_producer_loop,
                kwargs={
                    "worker_index": index,
                    "seed": seed,
                    "stockfish_path": str(stockfish_path),
                    "multipv_min": multipv_min,
                    "multipv_max": multipv_max,
                    "depth": depth,
                    "max_len": max_len,
                    "row_queue": self.row_queue,
                    "stop_event": self.stop_event,
                    "accepted_value": self.accepted_value,
                    "queue_depth_value": self.queue_depth_value,
                },
            )
            for index in range(num_producers)
        ]

    def start(self) -> None:
        """Start all producer processes."""
        for process in self.processes:
            process.start()

    def stop(self) -> None:
        """Stop all producer processes and join them."""
        self.stop_event.set()
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for process in self.processes:
                process.join(timeout=2.0)
        except KeyboardInterrupt:
            # Keep shutdown deterministic and let the parent continue cleanup.
            pass
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def queue_depth(self) -> int:
        """Return the current queue depth in rows."""
        with self.queue_depth_value.get_lock():
            return int(self.queue_depth_value.value)

    def accepted_rows(self) -> int:
        """Return the number of accepted rows queued by producers."""
        with self.accepted_value.get_lock():
            return int(self.accepted_value.value)

    def pop_rows(
        self, num_rows: int, *, timeout_s: float = 1.0
    ) -> list[dict[str, int | str | list[int] | None]]:
        """Pop rows from the queue, blocking briefly if needed."""
        rows: list[dict[str, int | str | list[int] | None]] = []
        for _ in range(num_rows):
            row = self.row_queue.get(timeout=timeout_s)
            rows.append(row)
            with self.queue_depth_value.get_lock():
                self.queue_depth_value.value -= 1
        return rows
