from pathlib import Path

import torch
import torch.nn.functional as F

from src.rl.config import RLPhase1Config
from src.rl.trainer import run_phase1_training
from src.tokenizer import Tokenizer


class ToyModel(torch.nn.Module):
    class config:
        block_size = 128

    def __init__(self, vocab_size: int, hidden_size: int = 16):
        super().__init__()
        self.config = ToyModel.config
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None, ignore_index=-100):
        logits = self.proj(self.embed(x))
        if y is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=ignore_index,
        )
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):  # noqa: ARG002
        return torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )


class StubDataSource:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def sample_prompt_batch(self, batch_size, device):
        prompt = torch.tensor(
            [[self.tokenizer.win_white_id, self.tokenizer.move_to_id["e2e4"]]],
            dtype=torch.long,
            device=device,
        )
        return prompt.repeat(batch_size, 1), torch.full(
            (batch_size,),
            2,
            dtype=torch.long,
            device=device,
        )

    def sample_supervised_batch(self, batch_size, device):
        rows = torch.tensor(
            [
                [
                    self.tokenizer.win_white_id,
                    self.tokenizer.move_to_id["e2e4"],
                    self.tokenizer.move_to_id["e7e5"],
                    self.tokenizer.eos_id,
                ]
            ],
            dtype=torch.long,
            device=device,
        ).repeat(batch_size, 1)
        return rows[:, :-1], rows[:, 1:]


class FakeArtifact:
    def __init__(self, name, type):  # noqa: A002
        self.name = name
        self.type = type
        self.logged_dirs = []

    def add_dir(self, path):
        self.logged_dirs.append(path)


class FakeWandb:
    def __init__(self):
        self.logs = []
        self.artifacts = []

    def log(self, payload):
        self.logs.append(payload)

    def Artifact(self, name, type):  # noqa: N802,A002
        artifact = FakeArtifact(name, type)
        self.artifacts.append(artifact)
        return artifact

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


def test_run_phase1_training_saves_timed_and_final_checkpoints(tmp_path):
    tokenizer = Tokenizer(Path("data/all_uci_moves.txt"))
    policy_model = ToyModel(tokenizer.get_vocab_size())
    reference_model = ToyModel(tokenizer.get_vocab_size())
    reference_model.load_state_dict(policy_model.state_dict())

    config = RLPhase1Config(
        batch_size=1,
        sft_batch_size=1,
        group_size=2,
        log_every=1,
        save_minutes=0.5,
        compile=False,
    )
    data_source = StubDataSource(tokenizer)
    wandb = FakeWandb()
    time_values = iter([0.0, 60.0, 60.0])

    result = run_phase1_training(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        data_source=data_source,
        config=config,
        artifact_dir=tmp_path,
        run_config={"stage": "test"},
        wandb_module=wandb,
        max_iters=1,
        indefinitely=False,
        checkpoint_source="dummy.pt",
        checkpoint_time_fn=lambda: next(time_values),
    )

    assert result["iter_num"] == 1
    assert (tmp_path / "model.pt").exists()
    checkpoint_dirs = sorted((tmp_path / "checkpoints").iterdir())
    assert any(path.name.startswith("timed_") for path in checkpoint_dirs)
    assert any(path.name.startswith("final_") for path in checkpoint_dirs)
    assert wandb.logs
    assert wandb.artifacts
