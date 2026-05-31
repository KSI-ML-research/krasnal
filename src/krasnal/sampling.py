"""Deterministic sampling shared by preprocessing and evaluation."""

from __future__ import annotations

from hashlib import blake2b


def sample_bool(
    seed: int,
    game_key: str,
    ply: int,
    probability: float,
) -> bool:
    """Deterministically sample True/False based on seed, game key, and ply."""
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    digest = blake2b(f"{seed}|{game_key}|{ply}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big") / 2**64
    return value < probability


def whats_on_square_index(
    *,
    game_key: str,
    ply: int,
    seed: int,
) -> int:
    """Pick square index 0..63 for a ``whats_on`` probe (deterministic).

    Uses per-game ``game_key`` (e.g. space-separated UCIs), ply, and seed so eval matches
    training without serializing board state during preprocessing.
    """
    h = blake2b(digest_size=16)
    h.update(b"krasnal.whats_on.v2")
    h.update(str(seed).encode())
    h.update(b"\0")
    h.update(game_key.encode())
    h.update(b"\0")
    h.update(str(ply).encode())
    d = h.digest()
    u = int.from_bytes(d[0:8], "big")
    return u % 64
