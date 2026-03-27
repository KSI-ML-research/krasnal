# Installation Guide

To set up the **Krasnal** chess engine for development, you need both Python (managed by `uv`) and Rust.

## Prerequisites

1. **Rust**: Install via [rustup](https://rustup.rs/).
   - Required for the data ingestion pipeline and high-performance components.
2. **uv**: Install via [astral.sh/uv](https://astral.sh/uv).
   - Used for Python dependency management and running scripts.

## Setup Steps

1. **Clone the repository**:
   ```bash
   git clone git@github.com:KSI-ML-research/krasnal.git
   cd krasnal
   ```

2. **Run the setup**:
   ```bash
   just setup
   ```

## Verification

Run the basic test suite to ensure everything is set up correctly:
```bash
uv run pytest
cargo test
```
