import os

import pytest
import torch

from src.rl import checkpoint as checkpoint_mod


def test_resolve_pretrained_checkpoint_accepts_explicit_file(tmp_path):
    model_path = tmp_path / "model.pt"
    torch.save({"x": 1}, model_path)

    assert (
        checkpoint_mod.resolve_pretrained_checkpoint(str(model_path), latest_pretrain=False)
        == model_path
    )


def test_resolve_pretrained_checkpoint_accepts_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_path = run_dir / "model.pt"
    torch.save({"x": 1}, model_path)

    assert (
        checkpoint_mod.resolve_pretrained_checkpoint(str(run_dir), latest_pretrain=False)
        == model_path
    )


def test_resolve_latest_pretrain_path_uses_newest_run(tmp_path, monkeypatch):
    pretrain_root = tmp_path / "pretrain"
    older = pretrain_root / "older"
    newer = pretrain_root / "newer"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    torch.save({"x": 1}, older / "model.pt")
    torch.save({"x": 1}, newer / "model.pt")
    monkeypatch.setattr(checkpoint_mod, "ARTIFACTS_DIR", tmp_path)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert checkpoint_mod.resolve_latest_pretrain_path() == newer / "model.pt"


def test_resolve_pretrained_checkpoint_rejects_invalid_source_combo():
    with pytest.raises(ValueError, match="exactly one"):
        checkpoint_mod.resolve_pretrained_checkpoint(None, latest_pretrain=False)

    with pytest.raises(ValueError, match="exactly one"):
        checkpoint_mod.resolve_pretrained_checkpoint("foo.pt", latest_pretrain=True)


def test_checkpoint_timer_triggers_and_resets():
    values = iter([0.0, 31.0, 31.0, 35.0])
    timer = checkpoint_mod.CheckpointTimer(30.0, time_fn=lambda: next(values))

    assert timer.should_save() is True
    timer.mark_saved()
    assert timer.should_save() is False
