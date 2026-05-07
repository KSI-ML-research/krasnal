import json
from importlib import util
from pathlib import Path

import torch

from krasnal.config import TrainConfig
from krasnal.trainer import DistributedInfo, save_model_state

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "training" / "sft_train.py"
_SPEC = util.spec_from_file_location("sft_train_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_run_config = _MODULE.build_run_config


class _DummyCfg:
    seed = 7
    cot_ratio = 0.25
    normal_dataset = "/tmp/normal.parquet"

    def get(self, key: str, default=None):
        values = {
            "piece_aware_moves": True,
            "side_prefixed_moves": False,
        }
        return values.get(key, default)

    def __init__(self) -> None:
        self.model = {"name": "tiny"}


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class _DummyMConf:
    block_size = 8
    n_layer = 1
    n_head = 1
    n_embd = 8
    dropout = 0.0
    bias = False


def test_save_model_state_copies_move_vocab(tmp_path):
    model = _DummyModel()
    path = tmp_path / "model.pt"
    source_vocab = tmp_path / "source-move_vocab.json"
    source_vocab.write_text(
        json.dumps(
            {
                "manifest": {
                    "piece_aware_moves": True,
                    "side_prefixed_moves": False,
                    "generation_timestamp": "test",
                    "vocab_size": 1,
                },
                "vocab": {"<pad>": 2},
            }
        )
    )

    save_model_state(model, path, move_vocab_path=source_vocab)

    assert path.exists()
    assert (tmp_path / "move_vocab.json").read_text() == source_vocab.read_text()


def test_build_run_config_includes_move_vocab_flags():
    cfg = _DummyCfg()
    tconf = TrainConfig(
        learning_rate=1e-3,
        min_lr=1e-5,
        epochs=1.0,
        warmup_iters=1,
        batch_size=4,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        grad_clip=0.0,
        log_interval=1,
        eval_interval=1,
        max_iters=1,
        steps_per_epoch=1,
    )
    mconf = _DummyMConf()

    run_config = build_run_config(
        cfg,
        tconf,
        mconf,
        cot_train_paths=[Path("a"), Path("b")],
        cot_eval_paths=[Path("c")],
        normal_dataset_path=Path(cfg.normal_dataset),
        vocab_size=32,
        total_iters=17,
        dist_info=DistributedInfo(False, 0, 1, 0),
    )

    assert run_config["piece_aware_moves"] is True
    assert run_config["side_prefixed_moves"] is False
    assert run_config["cot_train_shards"] == 2
    assert run_config["cot_eval_shards"] == 1
    assert run_config["move_vocab_path"].endswith("move_vocab.json")
