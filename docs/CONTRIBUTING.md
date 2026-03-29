# Contributing Guide

Contributions to **Krasnal** are welcome! Please follow these standards to ensure code quality and consistency.

## Tech Stack

- **Python**: Model training, UCI implementation, data processing, training pipeline.
- **DuckDB + Aix extension**: Data filtering and querying from Lichess database.
- **uv**: Project and dependency management.

## Code Quality Standards

We use `pre-commit` hooks to maintain high standards for all contributions.

### Python Requirements
- **Linter & Formatter**: [Ruff](https://github.com/astral-sh/ruff) (enforced via pre-commit).
- **Style**: Adhere to PEP 8, but prioritize Ruff's configuration.

## Pre-commit Hooks

Before committing, the following checks are run automatically on the files you've changed:
- Ruff check and format (Python).
- Gitleaks (Secret detection).
- Python tests (Pytest).

### Running Manually

If you want to run all checks on all files before committing:
```bash
uv run pre-commit run --all-files
```

## Pull Request Process

1. Create a new branch for your feature or bugfix.
2. Ensure all tests and pre-commit hooks pass.
3. Submit a Pull Request with a clear description of your changes.
