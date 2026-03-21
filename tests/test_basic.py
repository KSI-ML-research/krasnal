import os


def test_imports():
    """Test that core modules and scripts exist in their respective directories."""
    base_path = os.path.join(os.path.dirname(__file__), "..")

    src_path = os.path.join(base_path, "src")
    required_src_files = [
        "config.py",
        "dataset.py",
        "tokenizer.py",
        "model.py",
    ]

    for file in required_src_files:
        file_path = os.path.join(src_path, file)
        assert os.path.exists(file_path), f"Missing {file} in src/"

    required_inference_files = [
        "inference/__init__.py",
        "inference/abstracts.py",
        "inference/sampler.py",
        "inference/session.py",
        "inference/generator.py",
    ]

    for rel_path in required_inference_files:
        file_path = os.path.join(src_path, rel_path)
        assert os.path.exists(file_path), f"Missing {rel_path} in src/"

    scripts_path = os.path.join(base_path, "scripts")
    required_script_paths = [
        "preprocess.py",
        "pretrain.py",
    ]

    for rel_path in required_script_paths:
        file_path = os.path.join(scripts_path, rel_path)
        assert os.path.exists(file_path), f"Missing {rel_path} in scripts/"

    evals_path = os.path.join(base_path, "src", "evals")
    required_evals_files = [
        "__init__.py",
        "abstracts.py",
        "evaluator.py",
        "loss.py",
        "reporting.py",
        "run.py",
    ]

    for file in required_evals_files:
        file_path = os.path.join(evals_path, file)
        assert os.path.exists(file_path), f"Missing {file} in src/evals/"

    evals_metrics_path = os.path.join(evals_path, "metrics")
    required_evals_metrics_files = [
        "__init__.py",
        "top1_legal.py",
        "illegal_mass.py",
        "acpl.py",
        "cot_acpl.py",
    ]

    for file in required_evals_metrics_files:
        file_path = os.path.join(evals_metrics_path, file)
        assert os.path.exists(file_path), f"Missing {file} in src/evals/metrics/"
