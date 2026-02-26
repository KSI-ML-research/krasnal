import os


def test_imports():
    """Test that Python modules exist in src directory."""
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    required_files = [
        "config.py",
        "dataset.py",
        "preprocess.py",
        "tokenizer.py",
        "train.py",
    ]

    for file in required_files:
        file_path = os.path.join(src_path, file)
        assert os.path.exists(file_path), f"Missing {file} in src/"
