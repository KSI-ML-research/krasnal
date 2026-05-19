"""Build ``ChessModelProvider`` from env (split from ``run.py`` to avoid import cycles)."""

from __future__ import annotations

import os
from pathlib import Path

from krasnal.uci_engine.provider import ChessModelProvider, ModelProvider, RandomMockProvider


def build_provider() -> tuple[ChessModelProvider, str]:
    """Return ``(provider, engine_display_name)`` for the UCI session."""
    requested_provider = os.environ.get("KRASNAL_ENGINE_PROVIDER", "auto").strip().lower()

    if requested_provider == "mock":
        return RandomMockProvider(), "Krasnal Mock"

    artifact_dir_env = os.environ.get("KRASNAL_MODEL_ARTIFACT_DIR")
    if not artifact_dir_env:
        raise ValueError("Either set KRASNAL_ENGINE_PROVIDER=mock or KRASNAL_MODEL_ARTIFACT_DIR")
    artifact_dir = Path(artifact_dir_env)
    provider = ModelProvider.from_artifact_dir(artifact_dir)
    return provider, provider.engine_name
