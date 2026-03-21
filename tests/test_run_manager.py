from dataclasses import dataclass

from run_manager import _dataclass_to_dict, compute_run_hash


@dataclass
class MockModelConfig:
    n_layer: int = 6
    n_head: int = 6


@dataclass
class MockTrainConfig:
    learning_rate: float = 5e-4
    batch_size: int = 32


def test_compute_run_hash_deterministic():
    h1 = compute_run_hash(
        stage="pretrain",
        model_config=MockModelConfig(),
        train_config=MockTrainConfig(),
        seed=42,
        model_repr="tiny",
        dataset_mtime=1234567890,
    )
    h2 = compute_run_hash(
        stage="pretrain",
        model_config=MockModelConfig(),
        train_config=MockTrainConfig(),
        seed=42,
        model_repr="tiny",
        dataset_mtime=1234567890,
    )
    assert h1 == h2
    assert len(h1) == 8


def test_compute_run_hash_different_inputs_different_hashes():
    base = dict(
        stage="pretrain",
        model_config=MockModelConfig(),
        train_config=MockTrainConfig(),
        seed=42,
        model_repr="tiny",
        dataset_mtime=1234567890,
    )
    h0 = compute_run_hash(**base)

    assert compute_run_hash(**{**base, "stage": "finetune"}) != h0
    assert compute_run_hash(**{**base, "seed": 99}) != h0
    assert compute_run_hash(**{**base, "model_repr": "big"}) != h0
    assert compute_run_hash(**{**base, "dataset_mtime": 999}) != h0
    assert compute_run_hash(**{**base, "model_config": MockModelConfig(n_layer=12)}) != h0
    assert compute_run_hash(**{**base, "train_config": MockTrainConfig(batch_size=64)}) != h0


def test_dataclass_to_dict_nested():
    @dataclass
    class Inner:
        foo: int = 1
        bar: list[int] = (1, 2)

    @dataclass
    class Outer:
        x: Inner = None
        y: dict = None

    outer = Outer(x=Inner(), y={"a": [1, 2], "b": {"c": 3}})

    result = _dataclass_to_dict(outer)
    assert result == {"x": {"foo": 1, "bar": [1, 2]}, "y": {"a": [1, 2], "b": {"c": 3}}}


def test_dataclass_to_dict_primitive():
    assert _dataclass_to_dict("hello") == "hello"
    assert _dataclass_to_dict(42) == 42
    assert _dataclass_to_dict([1, 2, 3]) == [1, 2, 3]
