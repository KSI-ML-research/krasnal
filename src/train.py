import math
from contextlib import nullcontext

import torch
from datetime import datetime
from tqdm.auto import tqdm
from model import GPT, GPTConfig
from dataset import ChessDataset, collate_fn
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from config import (
    ChessGPTConfig,
    TrainConfig,
    MOVES_FILE,
    DATASET_PATH,
    MODEL_PATH,
    PAD_ID,
)


torch.manual_seed(42)


def get_lr(it, train_config):
    """Cosine annealing learning rate schedule with warmup."""
    # 1) linear warmup for warmup_iters steps
    if it < train_config.warmup_iters:
        return train_config.learning_rate * it / train_config.warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > train_config.max_iters:
        return train_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    assert train_config.max_iters > train_config.warmup_iters
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.max_iters - train_config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)


def main():
    print("Loading dataset...")
    train_dataset = ChessDataset(DATASET_PATH)
    tokenizer = Tokenizer(MOVES_FILE)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    print(f"Dataset size: {len(train_dataset)} games")

    # --- Model config ---
    mconf = ChessGPTConfig()
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

    # --- Training config ---
    tconf = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"Using device: {device}")

    # mixed precision context
    # use bfloat16 if available (Ampere+), otherwise float16
    if device_type == "cuda":
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    else:
        dtype = "float32"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    print(f"Mixed precision: {dtype}")

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay=tconf.weight_decay,
        learning_rate=tconf.learning_rate,
        betas=(tconf.beta1, tconf.beta2),
        device_type=device_type,
    )

    model.to(device)

    # torch.compile
    if tconf.compile and device_type == "cuda":
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, dynamic=False)

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=tconf.batch_size,
        num_workers=tconf.num_workers,
        collate_fn=collate_fn,
    )

    model.train()
    print("Starting training...")

    log_interval = 10
    iter_num = 0
    steps_per_epoch = len(train_loader)
    est_epochs = tconf.max_iters / max(steps_per_epoch, 1)
    pbar = tqdm(
        total=tconf.max_iters,
        desc=f"train (~{est_epochs:.2f} ep)",
        unit="iter",
        dynamic_ncols=True,
    )

    while iter_num < tconf.max_iters:
        for x, y in train_loader:
            # update learning rate for this iteration
            lr = get_lr(iter_num, tconf)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # forward pass with mixed precision
            with ctx:
                _, loss = model(x, y, ignore_index=PAD_ID)

            # backward pass with gradient scaling
            scaler.scale(loss).backward()

            # clip gradients
            if tconf.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tconf.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if iter_num % log_interval == 0:
                epoch_float = iter_num / max(steps_per_epoch, 1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr:.2e}",
                    epoch=f"{epoch_float:.2f}",
                )

            pbar.update(1)
            iter_num += 1
            if iter_num >= tconf.max_iters:
                break

    pbar.close()
    print("Training finished.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    params = (
        f"L{mconf.n_layer}_H{mconf.n_head}_E{mconf.n_embd}_I{tconf.max_iters}_B{tconf.batch_size}"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # unwrap compiled model if needed
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    history_path = MODEL_PATH.parent / f"final_{params}_{timestamp}.pt"
    torch.save(raw_model.state_dict(), history_path)
    torch.save(raw_model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} and {history_path}")


if __name__ == "__main__":
    main()
