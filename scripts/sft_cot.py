import argparse
import json
import queue
from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm
from utils import print_model_config, set_seed

import wandb
from config import (
    ARTIFACTS_DIR,
    EVAL_DATASET_PATH,
    MOVES_FILE,
    PRETRAIN_DATASET_PATH,
    SFT_COT_SHARDS_DIR,
    ChessGPTConfig,
    TrainConfig,
)
from dataset import ChessDataset, collate_fn
from model import GPT, GPTConfig
from rl.checkpoint import CheckpointTimer, resolve_pretrained_checkpoint, save_checkpoint
from sft import (
    CotProducerPool,
    CotReplaySource,
    CotShardWriter,
    RandomTokenSource,
    compute_batch_sizes,
    compute_split_losses,
    resolve_shard_paths,
)
from tokenizer import PAD_ID, Tokenizer
from trainer import cosine_warmup_lr, save_model_state, setup_runtime, unwrap_model

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Online-first CoT SFT from a pretrained checkpoint."
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--latest-pretrain", action="store_true")
    parser.add_argument(
        "--cot-shards-dir", type=Path, nargs="?", const=SFT_COT_SHARDS_DIR, default=None
    )
    parser.add_argument("--stockfish-path", type=Path, default=None)
    parser.add_argument("--normal-dataset", type=Path, default=PRETRAIN_DATASET_PATH)
    parser.add_argument("--cot-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="krasnal")
    parser.add_argument("--multipv-min", type=int, default=1)
    parser.add_argument("--multipv-max", type=int, default=3)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--movetime-ms", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--indefinitely", action="store_true")
    parser.add_argument("--save-minutes", type=float, default=30.0)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--save-cot-dir", type=Path, default=None)
    parser.add_argument("--num-producers", type=int, default=1)
    return parser.parse_args()


def build_model(tokenizer: Tokenizer) -> GPT:
    mconf = ChessGPTConfig()
    return GPT(
        GPTConfig(
            block_size=mconf.block_size,
            vocab_size=tokenizer.get_vocab_size(),
            n_layer=mconf.n_layer,
            n_head=mconf.n_head,
            n_embd=mconf.n_embd,
            dropout=mconf.dropout,
            bias=mconf.bias,
        )
    )


def load_state_dict(model_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def evaluate_loss(
    model, dataset_path: Path | list[Path], batch_size: int, num_workers: int, device: str
) -> float:
    eval_dataset = ChessDataset(dataset_path)
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        pin_memory=(device == "cuda"),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, loss = model(x, y, ignore_index=PAD_ID)
            valid_tokens = (y != PAD_ID).sum().item()
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
    model.train()
    if total_tokens == 0:
        raise ValueError("Eval dataset has no valid tokens")
    return total_loss / total_tokens


def build_lr_config(config: TrainConfig, max_iters: int | None, iter_num: int):
    effective_max_iters = max(max_iters or (iter_num + 2), config.warmup_iters + 1)

    class LRConfig:
        pass

    lr_config = LRConfig()
    lr_config.learning_rate = config.learning_rate
    lr_config.min_lr = config.min_lr
    lr_config.warmup_iters = config.warmup_iters
    lr_config.max_iters = effective_max_iters
    return lr_config


def validate_args(args) -> str:
    if (args.max_iters is None) == (not args.indefinitely):
        raise ValueError("Use exactly one of --max-iters or --indefinitely")
    if args.max_iters is not None and args.max_iters <= 0:
        raise ValueError("--max-iters must be > 0")
    if args.save_minutes <= 0:
        raise ValueError("--save-minutes must be > 0")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0")
    if args.num_producers <= 0:
        raise ValueError("--num-producers must be > 0")
    if not args.normal_dataset.exists():
        raise FileNotFoundError(f"Normal dataset not found at {args.normal_dataset}")
    if args.cot_shards_dir is not None:
        if args.depth is not None or args.movetime_ms is not None:
            raise ValueError("Replay mode does not accept search-budget generation flags")
        return "replay"
    if args.stockfish_path is None:
        raise ValueError("Online mode requires --stockfish-path")
    if (args.depth is None) == (args.movetime_ms is None):
        raise ValueError("Use exactly one of --depth or --movetime-ms")
    if not args.stockfish_path.exists():
        raise FileNotFoundError(f"Stockfish not found at {args.stockfish_path}")
    return "online"


def maybe_resolve_online_replay_paths(mode: str) -> list[Path]:
    if mode != "online":
        return []
    try:
        return resolve_shard_paths(SFT_COT_SHARDS_DIR)
    except FileNotFoundError:
        return []


def print_startup_summary(
    *,
    mode: str,
    num_producers: int,
    cot_batch_size: int,
    normal_batch_size: int,
    replay_paths: list[Path],
    save_cot_dir: Path,
    global_cot_dir: Path,
) -> None:
    replay_enabled = bool(replay_paths)
    print(
        f"Mode: {mode}  |  producers={num_producers}  |  cot_batch={cot_batch_size}  |  "
        f"normal_batch={normal_batch_size}\n"
        f"Replay first: {replay_enabled}  |  replay_shards={len(replay_paths)}\n"
        f"Run shards: {save_cot_dir}\n"
        f"Global shards: {global_cot_dir}"
    )


def main():
    args = parse_args()
    mode = validate_args(args)
    set_seed(args.seed)

    checkpoint_path = resolve_pretrained_checkpoint(args.model, args.latest_pretrain)
    tokenizer = Tokenizer(MOVES_FILE)
    model = build_model(tokenizer)
    model.load_state_dict(load_state_dict(checkpoint_path))

    device, device_type, dtype, ctx, scaler = setup_runtime()
    model.to(device)

    tconf = TrainConfig(learning_rate=5e-5)
    cot_batch_size, normal_batch_size = compute_batch_sizes(tconf.batch_size, args.cot_ratio)
    normal_source = RandomTokenSource(args.normal_dataset, seed=args.seed)

    optimizer = model.configure_optimizers(
        weight_decay=tconf.weight_decay,
        learning_rate=tconf.learning_rate,
        betas=(tconf.beta1, tconf.beta2),
        device_type=device_type,
    )

    if tconf.compile and device_type == "cuda":
        model = torch.compile(model, fullgraph=True, dynamic=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_DIR / "sft_cot" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_cot_dir = args.save_cot_dir or (artifact_dir / "cot_shards")
    global_cot_dir = SFT_COT_SHARDS_DIR
    manifest_path = artifact_dir / "cot_manifest.json"
    params_m = model.get_num_params() / 1_000_000

    replay_paths = maybe_resolve_online_replay_paths(mode)
    queue_max_rows = max(cot_batch_size * 16, tconf.batch_size * 8)

    run_config = {
        "stage": "sft_cot",
        "mode": mode,
        "seed": args.seed,
        "checkpoint_path": str(checkpoint_path),
        "normal_dataset": str(args.normal_dataset),
        "cot_shards_dir": str(args.cot_shards_dir) if args.cot_shards_dir else None,
        "stockfish_path": str(args.stockfish_path) if args.stockfish_path else None,
        "cot_ratio": args.cot_ratio,
        "batch_size": tconf.batch_size,
        "cot_batch_size": cot_batch_size,
        "normal_batch_size": normal_batch_size,
        "learning_rate": tconf.learning_rate,
        "multipv_min": args.multipv_min,
        "multipv_max": args.multipv_max,
        "depth": args.depth,
        "movetime_ms": args.movetime_ms,
        "max_iters": args.max_iters,
        "indefinitely": args.indefinitely,
        "save_minutes": args.save_minutes,
        "shard_size": args.shard_size,
        "save_cot_dir": str(save_cot_dir),
        "global_cot_dir": str(global_cot_dir),
        "num_producers": args.num_producers,
        "replay_enabled": bool(replay_paths),
        "replay_shard_count": len(replay_paths),
        "queue_max_rows": queue_max_rows,
        "dtype": dtype,
    }

    wandb.init(project=args.wandb_project, config=run_config)
    run_id = wandb.run.id  # type: ignore[union-attr]
    entity = wandb.run.entity  # type: ignore[union-attr]
    project = wandb.run.project  # type: ignore[union-attr]
    wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

    print_model_config(
        stage="SFT CoT",
        params_m=params_m,
        dataset_size=tconf.batch_size,
        dataset_label="batch",
        config=ChessGPTConfig(),
        vocab_size=tokenizer.get_vocab_size(),
        device=device,
        dtype=dtype,
        compile_enabled=tconf.compile,
        artifact_dir=artifact_dir,
    )
    print_startup_summary(
        mode=mode,
        num_producers=args.num_producers,
        cot_batch_size=cot_batch_size,
        normal_batch_size=normal_batch_size,
        replay_paths=replay_paths,
        save_cot_dir=save_cot_dir,
        global_cot_dir=global_cot_dir,
    )

    writer = CotShardWriter(
        output_dir=save_cot_dir,
        manifest_path=manifest_path,
        shard_size=args.shard_size,
        metadata=run_config,
    )
    global_writer = CotShardWriter(
        output_dir=global_cot_dir,
        manifest_path=None,
        shard_size=args.shard_size,
        metadata=run_config,
        filename_prefix=f"{timestamp}_",
    )
    checkpoint_timer = CheckpointTimer(interval_seconds=args.save_minutes * 60.0)
    total_iters = args.max_iters if args.max_iters is not None else None

    producer_pool = None
    replay_source = (
        CotReplaySource(args.cot_shards_dir, seed=args.seed + 1) if mode == "replay" else None
    )
    warm_replay_source = (
        CotReplaySource(SFT_COT_SHARDS_DIR, seed=args.seed + 2) if replay_paths else None
    )
    if mode == "online":
        producer_pool = CotProducerPool(
            num_producers=args.num_producers,
            queue_max_rows=queue_max_rows,
            stockfish_path=args.stockfish_path,
            multipv_min=args.multipv_min,
            multipv_max=args.multipv_max,
            depth=args.depth,
            movetime_ms=args.movetime_ms,
            seed=args.seed,
        )
        producer_pool.start()

    iter_num = 0
    interrupted = False
    wait_count = 0
    underflow_count = 0
    consumed_replay_rows = 0
    consumed_fresh_rows = 0
    latest_metrics: dict[str, float | int | str | bool | None] = {}
    progress = tqdm(total=total_iters, desc="sft-cot", unit="iter", dynamic_ncols=True)
    replay_progress = (
        tqdm(
            total=warm_replay_source.total_rows,
            desc="replay",
            unit="row",
            leave=False,
            dynamic_ncols=True,
        )
        if mode == "online" and warm_replay_source is not None
        else None
    )

    try:
        while args.indefinitely or (args.max_iters is not None and iter_num < args.max_iters):
            lr = cosine_warmup_lr(iter_num, build_lr_config(tconf, args.max_iters, iter_num))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            phase = "replay" if mode == "online" and warm_replay_source else "online"
            if mode == "replay":
                phase = "replay_only"
            if mode == "online":
                replay_batch = (
                    warm_replay_source.take_sequences(cot_batch_size) if warm_replay_source else []
                )
                consumed_replay_rows += len(replay_batch)
                if replay_progress is not None and replay_batch:
                    replay_progress.update(len(replay_batch))
                if warm_replay_source is not None and warm_replay_source.remaining_rows() == 0:
                    warm_replay_source = None
                    if replay_progress is not None:
                        replay_progress.close()
                        replay_progress = None
                fresh_rows: list[dict[str, int | str | list[int] | None]] = []
                fresh_target = cot_batch_size - len(replay_batch)
                while producer_pool is not None and len(fresh_rows) < fresh_target:
                    try:
                        fresh_rows.extend(
                            producer_pool.pop_rows(fresh_target - len(fresh_rows), timeout_s=1.0)
                        )
                    except queue.Empty:
                        wait_count += 1
                        underflow_count += 1
                fresh_batch = [
                    torch.tensor(row["token_ids"], dtype=torch.long) for row in fresh_rows
                ]
                consumed_fresh_rows += len(fresh_rows)
                if fresh_rows:
                    writer.add_rows(fresh_rows)
                    global_writer.add_rows(fresh_rows)
                cot_batch = replay_batch + fresh_batch
            else:
                cot_batch = replay_source.sample_sequences(cot_batch_size)
                consumed_replay_rows += len(cot_batch)

            normal_batch = normal_source.sample_sequences(normal_batch_size)
            batch = cot_batch + normal_batch
            source_ids = torch.tensor(
                [1] * len(cot_batch) + [0] * len(normal_batch), dtype=torch.long
            )
            permutation = torch.randperm(len(batch))
            batch = [batch[idx] for idx in permutation.tolist()]
            source_ids = source_ids[permutation].to(device, non_blocking=True)
            x, y = collate_fn(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                logits, loss = model(x, y, ignore_index=PAD_ID)

            scaler.scale(loss).backward()
            if tconf.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tconf.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            cot_loss, normal_loss = compute_split_losses(logits.detach(), y, source_ids)
            latest_metrics = {
                "iter": iter_num,
                "lr": lr,
                "phase": phase,
                "train_loss": float(loss.item()),
                "train_loss_cot": cot_loss,
                "train_loss_normal": normal_loss,
                "queue_depth": producer_pool.queue_depth() if producer_pool else 0,
                "generated_attempts": producer_pool.attempts() if producer_pool else None,
                "accepted_rows": producer_pool.accepted_rows() if producer_pool else None,
                "consumed_replay_rows": consumed_replay_rows,
                "consumed_fresh_rows": consumed_fresh_rows,
                "underflow_count": underflow_count,
                "wait_count": wait_count,
                "saved_rows": writer.total_rows,
                "buffered_rows": writer.buffered_rows(),
            }

            if iter_num % 10 == 0:
                wandb.log(latest_metrics)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    saved_rows=writer.total_rows,
                    queue=latest_metrics["queue_depth"],
                    phase=phase,
                )

            if checkpoint_timer.should_save():
                writer.flush()
                global_writer.flush()
                checkpoint_dir = save_checkpoint(
                    unwrap_model(model),
                    tokenizer=tokenizer,
                    checkpoint_root=artifact_dir / "checkpoints",
                    iter_num=iter_num,
                    kind="timed",
                    metadata=latest_metrics,
                )
                checkpoint_timer.mark_saved()
                artifact = wandb.Artifact(f"sft_cot_iter_{iter_num}", type="model")
                artifact.add_dir(str(checkpoint_dir))
                wandb.log_artifact(artifact)

            iter_num += 1
            progress.update(1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        progress.close()
        if replay_progress is not None:
            replay_progress.close()
        if producer_pool is not None:
            try:
                producer_pool.stop()
            except KeyboardInterrupt:
                # Continue to final flush/checkpoint when users press Ctrl-C repeatedly.
                pass

    writer.flush()
    global_writer.flush()

    model_path = artifact_dir / "model.pt"
    save_model_state(unwrap_model(model), model_path, tokenizer=tokenizer)
    save_checkpoint(
        unwrap_model(model),
        tokenizer=tokenizer,
        checkpoint_root=artifact_dir / "checkpoints",
        iter_num=iter_num,
        kind="final",
        metadata={"interrupted": interrupted, **latest_metrics},
    )

    normal_eval_loss = evaluate_loss(
        unwrap_model(model),
        EVAL_DATASET_PATH,
        tconf.batch_size,
        tconf.num_workers,
        device,
    )
    cot_eval_loss = None
    if mode == "replay":
        cot_eval_loss = evaluate_loss(
            unwrap_model(model),
            replay_source.shard_paths,
            tconf.batch_size,
            tconf.num_workers,
            device,
        )
    else:
        try:
            shard_paths = resolve_shard_paths(save_cot_dir)
        except FileNotFoundError:
            shard_paths = []
        if shard_paths:
            cot_eval_loss = evaluate_loss(
                unwrap_model(model),
                shard_paths,
                tconf.batch_size,
                tconf.num_workers,
                device,
            )

    eval_metrics = {"eval_loss_normal": normal_eval_loss}
    if cot_eval_loss is not None:
        eval_metrics["eval_loss_cot"] = cot_eval_loss
    wandb.log(eval_metrics)

    with open(artifact_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    with open(artifact_dir / "wandb_run_link.txt", "w") as f:
        f.write(f"{wandb_run_url}\n")

    artifact = wandb.Artifact("sft_cot", type="model")
    artifact.add_dir(str(artifact_dir))
    wandb.log_artifact(artifact)
    wandb.finish()

    print(f"Saved SFT CoT model to {model_path}")
    print(f"Saved CoT shards to {save_cot_dir}")
    print(f"Saved global CoT shards to {global_cot_dir}")
    print(f"eval_loss_normal={normal_eval_loss:.6f}")
    if cot_eval_loss is not None:
        print(f"eval_loss_cot={cot_eval_loss:.6f}")


if __name__ == "__main__":
    main()
