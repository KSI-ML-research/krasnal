import pytest

from krasnal.utils import (
    gpt_config_from_artifact_dict,
    read_model_config_json,
    write_artifact_config_json,
)


def test_write_artifact_config_json_requires_inference_keys(tmp_path):
    with pytest.raises(ValueError, match="missing inference keys"):
        write_artifact_config_json(tmp_path, {"block_size": 128})


def test_write_artifact_config_json_roundtrip(tmp_path):
    cfg = {
        "block_size": 128,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 64,
        "vocab_size": 9000,
        "dropout": 0.0,
        "use_time_conditioning": False,
        "time_conditioning_hidden": 1,
        "extra_metadata": True,
    }
    write_artifact_config_json(tmp_path, cfg)
    g = gpt_config_from_artifact_dict(read_model_config_json(tmp_path / "config.json"))
    assert g.n_layer == 2
    assert g.block_size == 128
