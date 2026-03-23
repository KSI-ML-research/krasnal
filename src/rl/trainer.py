from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from tokenizer import PAD_ID
from trainer import cosine_warmup_lr, save_model_state, setup_runtime

from .checkpoint import CheckpointTimer, save_checkpoint
from .config import RLPhase1Config
from .data import RLPhase1DataSource
from .losses import compute_grpo_loss, compute_masked_log_probs, compute_reference_kl
from .reward import score_phase1_rollouts
from .rollout import Phase1RolloutGenerator


def run_phase1_training(
    *,
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer,
    data_source: RLPhase1DataSource,
    config: RLPhase1Config,
    artifact_dir: Path,
    run_config: dict[str, Any],
    wandb_module,
    max_iters: int | None,
    indefinitely: bool,
    checkpoint_source: str,
    checkpoint_time_fn=None,
) -> dict[str, Any]:
    config.validate()
    device, device_type, dtype, ctx, scaler = setup_runtime()

    policy_model.to(device)
    reference_model.to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    train_model = policy_model
    if config.compile and device_type == "cuda":
        train_model = torch.compile(
            policy_model,
            mode=config.compile_mode,
            dynamic=config.compile_dynamic,
        )

    optimizer = policy_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = artifact_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_timer = CheckpointTimer(
        interval_seconds=config.save_minutes * 60.0,
        time_fn=checkpoint_time_fn,
    )
    rollout_generator = Phase1RolloutGenerator(
        policy_model,
        tokenizer,
        device=device,
        temperature=config.temperature,
    )

    iter_num = 0
    interrupted = False
    latest_metrics: dict[str, Any] = {}
    total_iters = max_iters if max_iters is not None else None
    progress = tqdm(total=total_iters, desc="rlvr_phase1", unit="iter", dynamic_ncols=True)

    try:
        while indefinitely or (max_iters is not None and iter_num < max_iters):
            lr = cosine_warmup_lr(iter_num, _build_lr_config(config, max_iters, iter_num))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            prompts, prompt_lengths = data_source.sample_prompt_batch(config.batch_size, device)
            rollouts = rollout_generator.generate(
                prompts,
                prompt_lengths,
                group_size=config.group_size,
                think_min_tokens=config.think_min_tokens,
                think_max_tokens=config.think_max_tokens,
            )
            rewards, reward_metrics = score_phase1_rollouts(
                rollouts.token_ids,
                rollouts.prompt_lengths,
                rollouts.think_lengths,
                tokenizer,
            )
            rewards = rewards.to(device)

            x_sft, y_sft = data_source.sample_supervised_batch(config.sft_batch_size, device)

            train_model.train()
            with ctx:
                seq_log_probs, token_counts = compute_masked_log_probs(
                    train_model, rollouts.token_ids, rollouts.completion_mask
                )
                rl_loss = compute_grpo_loss(
                    seq_log_probs,
                    token_counts,
                    rewards,
                    group_size=config.group_size,
                )
                kl_loss = compute_reference_kl(
                    train_model,
                    reference_model,
                    rollouts.token_ids,
                    rollouts.completion_mask,
                )
                _, sft_loss = train_model(x_sft, y_sft, ignore_index=PAD_ID)
                total_loss = rl_loss + (config.kl_coef * kl_loss) + (config.sft_coef * sft_loss)

            scaler.scale(total_loss).backward()
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            latest_metrics = {
                "iter": iter_num,
                "lr": lr,
                "total_loss": float(total_loss.item()),
                "rl_loss": float(rl_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "sft_loss": float(sft_loss.item()),
                "reward_mean": reward_metrics["reward_mean"],
                "legal_thinking_ratio": reward_metrics["legal_thinking_ratio"],
                "played_move_legal_rate": reward_metrics["played_move_legal_rate"],
                "think_tokens_mean": float(rollouts.think_lengths.float().mean().item()),
            }
            if iter_num % config.log_every == 0:
                wandb_module.log(latest_metrics)
                progress.set_postfix(
                    loss=f"{latest_metrics['total_loss']:.4f}",
                    reward=f"{latest_metrics['reward_mean']:.3f}",
                )

            if checkpoint_timer.should_save():
                checkpoint_dir = save_checkpoint(
                    policy_model,
                    tokenizer=tokenizer,
                    checkpoint_root=checkpoints_dir,
                    iter_num=iter_num,
                    kind="timed",
                    metadata={"checkpoint_source": checkpoint_source, **latest_metrics},
                )
                checkpoint_timer.mark_saved()
                artifact = wandb_module.Artifact(f"rlvr_phase1_iter_{iter_num}", type="model")
                artifact.add_dir(str(checkpoint_dir))
                wandb_module.log_artifact(artifact)

            iter_num += 1
            progress.update(1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        progress.close()

    save_model_state(policy_model, artifact_dir / "model.pt", tokenizer=tokenizer)
    save_checkpoint(
        policy_model,
        tokenizer=tokenizer,
        checkpoint_root=checkpoints_dir,
        iter_num=iter_num,
        kind="final",
        metadata={
            "interrupted": interrupted,
            "checkpoint_source": checkpoint_source,
            **latest_metrics,
        },
    )

    with open(artifact_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    artifact = wandb_module.Artifact("rlvr_phase1", type="model")
    artifact.add_dir(str(artifact_dir))
    wandb_module.log_artifact(artifact)

    return {
        "iter_num": iter_num,
        "interrupted": interrupted,
        **latest_metrics,
    }


def _build_lr_config(config: RLPhase1Config, max_iters: int | None, iter_num: int):
    effective_max_iters = max(max_iters or (iter_num + 2), config.warmup_iters + 1)

    class LRConfig:
        pass

    lr_config = LRConfig()
    lr_config.learning_rate = config.learning_rate
    lr_config.min_lr = config.min_lr
    lr_config.warmup_iters = config.warmup_iters
    lr_config.max_iters = effective_max_iters
    return lr_config
