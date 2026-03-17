import os


def test_imports():
    """Test that essential Python modules exist in src and scripts directories."""
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    required_files = [
        "config.py",
        "dataset.py",
        "tokenizer.py",
    ]

    for file in required_files:
        file_path = os.path.join(src_path, file)
        assert os.path.exists(file_path), f"Missing {file} in src/"
    scripts_path = os.path.join(os.path.dirname(__file__), "..", "scripts")
    required_scripts = [
        "preprocess.py",
        "pretrain.py",
    ]

    for file in required_scripts:
        file_path = os.path.join(scripts_path, file)
        assert os.path.exists(file_path), f"Missing {file} in scripts/"
