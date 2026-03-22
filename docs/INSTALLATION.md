# Installation Guide

To set up the **Krasnal** chess engine for development, you need both Python (managed by `uv`) and Rust.

## Prerequisites

1. **Rust**: Install via [rustup](https://rustup.rs/).
   - Required for the data ingestion pipeline and high-performance components.
2. **uv**: Install via [astral.sh/uv](https://astral.sh/uv).
   - Used for Python dependency management and running scripts.

## Setup Steps

1. **Clone the repository** (replace `<REPOSITORY_CLONE_URL>` with the HTTPS or SSH URL copied from the project's GitHub "Code" button):
   ```bash
   git clone <REPOSITORY_CLONE_URL>
   cd krasnal
   ```

2. **Install Python dependencies**:
   ```bash
   uv sync
   ```

3. **Install PyTorch variant (choose one)**:
   - CPU (recommended default):
   ```bash
   uv run python scripts/install_torch.py --target cpu
   ```
   - CUDA 12.4 (NVIDIA GPU):
   ```bash
   uv run python scripts/install_torch.py --target cu124
   ```

4. **Install Pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

5. **Build Rust components** (optional, handled by scripts if needed):
   ```bash
   cargo build --release
   ```

## Verification

Run the basic test suite to ensure everything is set up correctly:
```bash
uv run pytest
cargo test
```
