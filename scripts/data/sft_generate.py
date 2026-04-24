#!/usr/bin/env python3
"""Offline CoT shard generator driven by Stockfish."""

import queue
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from krasnal.sft import CotProducerPool, CotShardWriter, validate_stockfish
from krasnal.utils import set_seed

DEFAULT_SHARD_SIZE = 8192
DEFAULT_BATCH_SIZE = 32
DEFAULT_QUEUE_MULTIPLIER = 512


def resolve_max_len(cfg: DictConfig) -> int:
    if cfg.max_len is not None:
        return int(cfg.max_len)
    return int(cfg.model.block_size)


@hydra.main(version_base=None, config_path="../../config", config_name="sft_generate")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = DictConfig(cfg)

    if not cfg.indefinitely and (cfg.num_rows is None or int(cfg.num_rows) <= 0):
        raise ValueError("Either num_rows > 0 or indefinitely=true must be specified")

    if cfg.stockfish_path is None:
        raise ValueError("stockfish_path must be set (e.g. stockfish_path=/usr/bin/stockfish)")

    stockfish_path = Path(str(cfg.stockfish_path))
    depth = int(cfg.depth)
    validate_stockfish(stockfish_path, depth)

    seed = int(cfg.seed)
    set_seed(seed)

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_len = resolve_max_len(cfg)
    batch_size = int(cfg.batch_size)
    num_producers = int(cfg.num_producers)
    queue_max_rows = (
        int(cfg.queue_max_rows)
        if cfg.queue_max_rows is not None
        else max(batch_size * 8, num_producers * DEFAULT_QUEUE_MULTIPLIER)
    )

    writer = CotShardWriter(output_dir=output_dir, shard_size=int(cfg.shard_size))

    producer_pool = CotProducerPool(
        num_producers=num_producers,
        queue_max_rows=queue_max_rows,
        stockfish_path=stockfish_path,
        multipv_min=int(cfg.multipv_min),
        multipv_max=int(cfg.multipv_max),
        depth=depth,
        max_len=max_len,
        seed=seed,
    )
    producer_pool.start()

    target = int(cfg.num_rows) if cfg.num_rows is not None else None
    produced = 0
    progress = tqdm(total=target, desc="generate-cot", unit="row", dynamic_ncols=True)
    try:
        while bool(cfg.indefinitely) or (target is not None and produced < target):
            desired = batch_size if target is None else min(batch_size, target - produced)
            if desired <= 0:
                break
            try:
                rows = producer_pool.pop_rows(desired, timeout_s=1.0)
            except queue.Empty:
                continue
            if not rows:
                continue
            produced += len(rows)
            writer.add_rows(rows)
            progress.update(len(rows))
            if target is not None and produced >= target:
                break
    except KeyboardInterrupt:
        pass
    finally:
        progress.close()
        producer_pool.stop()
        writer.flush()

    print(f"Generated {produced} CoT rows into {output_dir}")


if __name__ == "__main__":
    main()
