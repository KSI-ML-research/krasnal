import argparse
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    MOVES_FILE,
    RLVR_DATASET_PATH,
    ChessGPTConfig,
    GRPOConfig,
    TrainConfig,
)
from src.dataset import ChessDataset, prompt_collate_fn
from src.model import GPT, GPTConfig
from src.rl import GRPOTrainer, compute_advantages, generate_rollouts
from src.run_manager import (
    compute_params_M,
    create_run_folder,
    find_latest_run,
    find_run_by_hash,
    save_run_config,
    update_run_config,
)
from src.tokenizer import SPECIAL_TOKENS, THINK_END_ID, THINK_START_ID, Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO legal thinking training")
    parser.add_argument(
        "--parent",
        type=str,
        default=None,
        help="Parent run hash or folder name (optional if --latest is used).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest SFT run as parent.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts per iteration")
    parser.add_argument("--group-size", type=int, default=8, help="Rollouts per prompt (G in GRPO)")
    parser.add_argument("--kl-coeff", type=float, default=0.01, help="KL penalty weight")
    parser.add_argument(
        "--think-min-tokens", type=int, default=8, help="Minimum tokens in think block"
    )
    parser.add_argument(
        "--think-max-tokens", type=int, default=32, help="Maximum tokens before forcing think_end"
    )
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Scale factor for rewards")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def compute_reward(completion_ids, board, actual_move_id, tokenizer):
    """
    Compute legal thinking reward.

    Reward:
        +1: thinking contains the actual move (legal move that matches the response)
        +0.5: thinking contains other legal moves (but not the actual one)
        -1: thinking contains illegal moves
         0: thinking is empty or malformed
    """
    special_ids = set(SPECIAL_TOKENS)

    has_start = THINK_START_ID in completion_ids
    has_end = THINK_END_ID in completion_ids

    if not (has_start and has_end):
        return 0.0

    start_idx = completion_ids.index(THINK_START_ID)
    end_idx = completion_ids.index(THINK_END_ID)

    if start_idx >= end_idx:
        return 0.0

    think_content = completion_ids[start_idx + 1 : end_idx]

    has_illegal = False
    has_actual_move = False
    has_other_legal = False

    actual_move_uci = tokenizer.id_to_move.get(actual_move_id, "")

    think_board = board.copy()
    for t_id in think_content:
        if t_id in special_ids:
            continue

        uci = tokenizer.id_to_move.get(t_id, "")
        if not uci:
            continue

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            has_illegal = True
            continue

        if move not in think_board.legal_moves:
            has_illegal = True
        else:
            if uci == actual_move_uci:
                has_actual_move = True
            else:
                has_other_legal = True
            think_board.push(move)

    if has_illegal:
        return -1.0
    if has_actual_move:
        return 1.0
    if has_other_legal:
        return 0.5
    return 0.0


def main():
    args = parse_args()
    stage_start = time.perf_counter()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    parent_run = None
    parent_hash = None
    if args.latest:
        parent_run = find_latest_run("sft")
        if parent_run:
            import json

            with (parent_run / "config.json").open() as f:
                parent_config = json.load(f)
            parent_hash = parent_config.get("run_hash")
            print(f"Using latest SFT run: {parent_run.name}")
        else:
            print("No SFT run found. Training from random weights.")
    elif args.parent:
        parent_run = find_run_by_hash(args.parent, stage="sft")
        if parent_run is None:
            parent_run = Path(args.parent)
        if (parent_run / "config.json").exists():
            import json

            with (parent_run / "config.json").open() as f:
                parent_config = json.load(f)
            parent_hash = parent_config.get("run_hash")
        print(f"Parent run: {parent_run.name}")

    mconf = ChessGPTConfig()
    gconf = GRPOConfig(
        group_size=args.group_size, num_samples=args.batch_size, kl_coeff=args.kl_coeff
    )
    tconf = TrainConfig()

    tokenizer = Tokenizer(MOVES_FILE)
    vocab_size = tokenizer.get_vocab_size()

    model_config = GPTConfig(
        block_size=mconf.block_size,
        vocab_size=vocab_size,
        n_layer=mconf.n_layer,
        n_head=mconf.n_head,
        n_embd=mconf.n_embd,
        dropout=mconf.dropout,
        bias=mconf.bias,
    )
    model = GPT(model_config)

    parent_model_path = parent_run / "model.pt" if parent_run else None
    if parent_model_path and parent_model_path.exists():
        print(f"Loading checkpoint from {parent_model_path}")
        model.load_state_dict(torch.load(parent_model_path, map_location="cpu", weights_only=True))
    else:
        if parent_run:
            print("WARNING: Parent model not found. Training from random weights.")
        else:
            print("No parent run specified. Training from random weights.")

    params_M = compute_params_M(model)
    run_folder, run_hash, commit_hash = create_run_folder(
        stage="grpo_legal",
        params_M=params_M,
        model_config=mconf,
        train_config=tconf,
        seed=args.seed,
        parent=parent_run.name if parent_run else None,
    )
    print(f"Run folder: {run_folder.name}")
    print(f"Run hash: {run_hash}")
    print(f"Git commit: {commit_hash}")
    print(
        f"layers={mconf.n_layer}, heads={mconf.n_head}, embd={mconf.n_embd}, "
        f"context={mconf.block_size}, vocab={vocab_size}"
    )

    save_run_config(
        folder=run_folder,
        stage="grpo_legal",
        run_hash=run_hash,
        params_M=params_M,
        model_config=mconf,
        train_config=tconf,
        seed=args.seed,
        commit_hash=commit_hash,
        parent=parent_run.name if parent_run else None,
        parent_hash=parent_hash,
        extra={
            "kl_coeff": args.kl_coeff,
            "group_size": args.group_size,
            "max_iters": args.max_iters,
            "think_min_tokens": args.think_min_tokens,
            "think_max_tokens": args.think_max_tokens,
            "dataset_path": str(RLVR_DATASET_PATH),
        },
    )

    model.to(device)

    ref_model = GPT(model_config)
    ref_model.load_state_dict(model.state_dict())
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    train_dataset = ChessDataset(RLVR_DATASET_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=prompt_collate_fn
    )

    trainer = GRPOTrainer(model, gconf)
    optimizer = model.configure_optimizers(
        weight_decay=tconf.weight_decay,
        learning_rate=tconf.learning_rate,
        betas=(tconf.beta1, tconf.beta2),
        device_type=device.type,
    )

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    print("Starting GRPO legal thinking training...")
    print(f"think_range=[{args.think_min_tokens}, {args.think_max_tokens}]")

    pbar = tqdm(total=args.max_iters)
    last_loss = None
    iter_num = 0

    while iter_num < args.max_iters:
        for prompts in train_loader:
            if iter_num >= args.max_iters:
                break

            prompts = prompts.to(device)
            model.eval()

            all_ids, completions_mask = generate_rollouts(
                model,
                prompts,
                args.group_size,
                tokenizer,
                think_min_tokens=args.think_min_tokens,
                think_max_tokens=args.think_max_tokens,
            )

            rewards = []
            for i in range(all_ids.size(0)):
                board = chess.Board()
                prompt_tokens = prompts[i // args.group_size]
                special_ids_set = set(SPECIAL_TOKENS)

                for t in prompt_tokens:
                    t_id = t.item()
                    if t_id not in special_ids_set:
                        move_uci = tokenizer.id_to_move.get(t_id, "")
                        if move_uci:
                            try:
                                board.push_uci(move_uci)
                            except Exception:
                                break

                completion_ids = all_ids[i, prompts.size(1) :].tolist()

                actual_move_id = None
                for t_id in completion_ids:
                    if t_id not in special_ids_set:
                        actual_move_id = t_id
                        break

                if actual_move_id is None:
                    reward = -1.0
                else:
                    reward = compute_reward(completion_ids, board, actual_move_id, tokenizer)

                rewards.append(reward * args.reward_scale)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
            advantages = compute_advantages(rewards_tensor, args.group_size)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            with ctx:
                logits, _ = model(all_ids, targets=all_ids)
                log_probs = F.log_softmax(logits, dim=-1)
                actual_log_probs = torch.gather(
                    log_probs[:, :-1, :], 2, all_ids[:, 1:, None]
                ).squeeze(-1)

                with torch.inference_mode():
                    ref_logits, _ = ref_model(all_ids, targets=all_ids)
                    ref_log_probs = torch.gather(
                        F.log_softmax(ref_logits[:, :-1, :], dim=-1), 2, all_ids[:, 1:, None]
                    ).squeeze(-1)

                kl = (
                    torch.exp(ref_log_probs - actual_log_probs)
                    - (ref_log_probs - actual_log_probs)
                    - 1
                )
                mask = completions_mask[:, 1:].to(device)
                batch_advantages = advantages.unsqueeze(1).repeat(1, actual_log_probs.size(1))

                loss = trainer.compute_loss(
                    actual_log_probs * mask,
                    ref_log_probs * mask,
                    batch_advantages * mask,
                    kl_penalty=(kl * mask).sum() / mask.sum(),
                )

            last_loss = float(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(
                loss=f"{last_loss:.4f}",
                avg_reward=f"{rewards_tensor.mean().item():.2f}",
            )
            pbar.update(1)
            iter_num += 1

            if iter_num % 100 == 0:
                rewards_min = rewards_tensor.min().item()
                rewards_mean = rewards_tensor.mean().item()
                rewards_max = rewards_tensor.max().item()
                print(
                    f"iter={iter_num} loss={last_loss:.4f} reward: "
                    f"min={rewards_min:.3f} mean={rewards_mean:.3f} max={rewards_max:.3f}"
                )

    pbar.close()

    torch.save(model.state_dict(), run_folder / "model.pt")
    print(f"Training complete. Model saved to {run_folder / 'model.pt'}")

    update_run_config(
        run_folder,
        {
            "final_loss": last_loss,
            "duration_seconds": time.perf_counter() - stage_start,
        },
    )


if __name__ == "__main__":
    from contextlib import nullcontext

    main()
