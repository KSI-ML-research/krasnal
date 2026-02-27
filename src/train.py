import torch
from datetime import datetime
from tqdm.auto import tqdm
from mingpt.model import GPT
from mingpt.trainer import Trainer
from dataset import ChessDataset, collate_fn
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from config import ChessGPTConfig, TrainConfig, MOVES_FILE, DATASET_PATH, MODEL_PATH


def main():
    print("Loading dataset...")
    train_dataset = ChessDataset(DATASET_PATH)
    tokenizer = Tokenizer(MOVES_FILE)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    print(f"Dataset size: {len(train_dataset)} games")

    mconf = ChessGPTConfig()
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.vocab_size = vocab_size
    model_config.block_size = mconf.block_size
    model_config.n_layer = mconf.n_layer
    model_config.n_head = mconf.n_head
    model_config.n_embd = mconf.n_embd

    model = GPT(model_config)

    tconf = TrainConfig()
    train_config = Trainer.get_default_config()
    train_config.learning_rate = tconf.learning_rate
    train_config.max_iters = tconf.max_iters
    train_config.batch_size = tconf.batch_size
    train_config.num_workers = tconf.num_workers
    train_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {train_config.device}")

    optimizer = model.configure_optimizers(train_config)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
    )

    model.to(train_config.device)
    model.train()

    print("Starting training...")

    log_interval = 10
    iter_num = 0
    steps_per_epoch = len(train_loader)
    est_epochs = train_config.max_iters / max(steps_per_epoch, 1)
    pbar = tqdm(
        total=train_config.max_iters,
        desc=f"train (~{est_epochs:.2f} ep)",
        unit="iter",
        dynamic_ncols=True,
    )
    while iter_num < train_config.max_iters:
        for x, y in train_loader:
            x = x.to(train_config.device)
            y = y.to(train_config.device)

            _, loss = model(x, y)

            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if iter_num % log_interval == 0:
                epoch_float = iter_num / max(steps_per_epoch, 1)
                pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch_float:.2f}")

            pbar.update(1)

            iter_num += 1
            if iter_num >= train_config.max_iters:
                break

    pbar.close()

    print("Training finished.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    params = (
        f"L{mconf.n_layer}_H{mconf.n_head}_E{mconf.n_embd}_I{tconf.max_iters}_B{tconf.batch_size}"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    history_path = MODEL_PATH.parent / f"final_{params}_{timestamp}.pt"
    torch.save(model.state_dict(), history_path)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} and {history_path}")


if __name__ == "__main__":
    main()
