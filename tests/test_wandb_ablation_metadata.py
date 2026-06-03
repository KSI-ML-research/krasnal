from krasnal.utils import ablation_metadata_from_env, ablation_tags


def test_ablation_metadata_from_env(monkeypatch):
    monkeypatch.setenv("RUN_NAME", "baseline_10000-gelu")
    monkeypatch.setenv("RUN_GROUP", "krasnal-medium-baseline_10000")
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_JOB_NAME", "krasnal-baseline_10000-gelu")
    monkeypatch.setenv("KRASNAL_TOKENIZED_DIR", "data/2_tokenized_ablations/baseline_10000")
    monkeypatch.setenv("WANDB_NAME", "baseline_10000-gelu-123456")
    monkeypatch.setenv("WANDB_RUN_GROUP", "krasnal-medium-baseline_10000")

    metadata = ablation_metadata_from_env()

    assert metadata == {
        "ablation_name": "baseline_10000-gelu",
        "ablation_group": "krasnal-medium-baseline_10000",
        "slurm_job_id": "123456",
        "slurm_job_name": "krasnal-baseline_10000-gelu",
        "tokenized_dir": "data/2_tokenized_ablations/baseline_10000",
        "wandb_name": "baseline_10000-gelu-123456",
        "wandb_group": "krasnal-medium-baseline_10000",
        "ablation_data_variant": "baseline_10000",
        "ablation_train_variant": "gelu",
    }
    assert ablation_tags(metadata) == ("ablation", "baseline_10000", "gelu")


def test_ablation_metadata_ignores_normal_training(monkeypatch):
    monkeypatch.delenv("RUN_NAME", raising=False)

    assert ablation_metadata_from_env() == {}
    assert ablation_tags({}) == ()
