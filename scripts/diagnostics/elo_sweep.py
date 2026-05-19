#!/usr/bin/env -S uv run python
"""Probe all Elo tokens to find max-contrast pair."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import torch

from krasnal.config import CLOCK_IGNORE_ID, MOVE_VOCAB_PATH
from krasnal.inference import Game, StatelessBatchInferenceSession, load_model
from krasnal.tokens import (
    ELO_1000_1099_ID,
    ELO_1100_1199_ID,
    ELO_1200_1299_ID,
    ELO_1300_1399_ID,
    ELO_1400_1499_ID,
    ELO_1500_1599_ID,
    ELO_1600_1699_ID,
    ELO_1700_1799_ID,
    ELO_1800_1899_ID,
    ELO_1900_1999_ID,
    ELO_2000_2099_ID,
    ELO_2100_2199_ID,
    ELO_ABOVE_2200_ID,
    ELO_BELOW_1000_ID,
    ID_TO_MOVE,
    TC_RAPID_INC_ID,
    WHITE_WON_ID,
    legal_token_ids,
    load_move_vocab,
    to_uci,
)
from krasnal.utils import (
    gpt_config_from_artifact_dict,
    read_model_config_json,
    resolve_runtime_device,
)

ARTIFACT_DIR = Path("artifacts/pretrain/20260515_163216")

ELO_BUCKETS: list[tuple[str, int]] = [
    ("<1000", ELO_BELOW_1000_ID),
    ("1000-1099", ELO_1000_1099_ID),
    ("1100-1199", ELO_1100_1199_ID),
    ("1200-1299", ELO_1200_1299_ID),
    ("1300-1399", ELO_1300_1399_ID),
    ("1400-1499", ELO_1400_1499_ID),
    ("1500-1599", ELO_1500_1599_ID),
    ("1600-1699", ELO_1600_1699_ID),
    ("1700-1799", ELO_1700_1799_ID),
    ("1800-1899", ELO_1800_1899_ID),
    ("1900-1999", ELO_1900_1999_ID),
    ("2000-2099", ELO_2000_2099_ID),
    ("2100-2199", ELO_2100_2199_ID),
    ("2200+", ELO_ABOVE_2200_ID),
]

TEST_POSITIONS = [
    {
        "label": "italian_two_knights",
        "description": "Italian Game, Two Knights Defense — after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
    },
]


def _token_label(tid: int) -> str:
    return ID_TO_MOVE.get(tid, f"id:{tid}")


def _make_clock_seq(n: int) -> list[int]:
    return [CLOCK_IGNORE_ID] * n


def main() -> None:
    import bulletchess as bc

    artifact_dir = ARTIFACT_DIR
    config_dict = read_model_config_json(artifact_dir / "config.json")
    gpt_config = gpt_config_from_artifact_dict(config_dict)
    piece_aware = bool(config_dict.get("piece_aware_moves", True))
    side_prefixed = bool(config_dict.get("side_prefixed_moves", True))
    load_move_vocab(
        MOVE_VOCAB_PATH,
        piece_aware_moves=piece_aware,
        side_prefixed_moves=side_prefixed,
    )

    device = resolve_runtime_device()
    print(f"Device: {device}")
    model = load_model(str(artifact_dir / "model.pt"), device, gpt_config)
    print(f"Model loaded: {gpt_config.n_layer}L, {gpt_config.n_head}H, {gpt_config.n_embd}E")

    use_time = gpt_config.use_time_conditioning
    session = StatelessBatchInferenceSession(model, device)

    for pos in TEST_POSITIONS:
        moves = pos["moves"]
        _fen_from_moves(moves)
        print(f"\n=== {pos['label']} ===")

        # Legal moves
        ref_board = bc.Board()
        for uci in moves:
            ref_board.apply(bc.Move.from_uci(uci))
        legal = legal_token_ids(ref_board)
        legal_info = [
            {"token_id": tid, "token_label": _token_label(tid), "uci": to_uci(tid) or ""}
            for tid in sorted(legal)
        ]
        print(f"  Legal moves: {len(legal)}")

        # Build games: one per Elo bucket, fixed WHITE_WON outcome
        games: list[Game] = []
        bucket_info: list[dict] = []
        for label, elo_id in ELO_BUCKETS:
            game = Game(
                white_elo_token=elo_id,
                black_elo_token=elo_id,
                time_control_token=TC_RAPID_INC_ID,
                target_outcome_token=WHITE_WON_ID,
            )
            for uci in moves:
                game.feed_uci(uci)
            games.append(game)
            bucket_info.append({"bucket": label, "elo_id": elo_id})

        # Forward pass
        if use_time:
            clock_seqs = [_make_clock_seq(len(g.context_tokens())) for g in games]
            all_logits = session.get_legal_logits_batch(
                games,
                active_clock_sequences=clock_seqs,
                opponent_clock_sequences=clock_seqs,
            )
        else:
            all_logits = session.get_legal_logits_batch(games)

        all_probs = torch.softmax(all_logits, dim=-1)

        # Collect probs per bucket
        results: dict[str, list[float]] = {}
        for idx, info in enumerate(bucket_info):
            probs = all_probs[idx]
            probs_list = [float(probs[tid].item()) for tid in sorted(legal)]
            results[info["bucket"]] = probs_list

        # --- Analyze all pairs ---
        print("\n  === ALL PAIR COMPARISONS (TV distance) ===")
        pairs_list = []
        for (b1, p1), (b2, p2) in combinations(results.items(), 2):
            tv = 0.5 * sum(abs(a - b) for a, b in zip(p1, p2, strict=False))
            max_delta = max(abs(a - b) for a, b in zip(p1, p2, strict=False))
            max_move_idx = max(range(len(p1)), key=lambda i: abs(p1[i] - p2[i]))
            max_move_uci = legal_info[max_move_idx]["uci"]
            pairs_list.append((tv, b1, b2, max_delta, max_move_uci))

        pairs_list.sort(reverse=True)

        print(f"  {'Rank':<5} {'TV':<8} {'Bucket A':<15} {'Bucket B':<15} {'Max Δ':<8} {'Move'}")
        print(f"  {'-' * 60}")
        for rank, (tv, b1, b2, md, uci) in enumerate(pairs_list, 1):
            print(f"  {rank:<5} {tv:<8.4f} {b1:<15} {b2:<15} {md:<8.4f} {uci}")

        print("\n  === ONLY IN-DISTRIBUTION (≥1800-1899) ===")
        id_buckets = {
            k: v
            for k, v in results.items()
            if k in ("1800-1899", "1900-1999", "2000-2099", "2100-2199", "2200+")
        }
        id_pairs = []
        for (b1, p1), (b2, p2) in combinations(id_buckets.items(), 2):
            tv = 0.5 * sum(abs(a - b) for a, b in zip(p1, p2, strict=False))
            max_delta = max(abs(a - b) for a, b in zip(p1, p2, strict=False))
            id_pairs.append((tv, b1, b2, max_delta))
        id_pairs.sort(reverse=True)
        print(f"  {'Rank':<5} {'TV':<8} {'Bucket A':<15} {'Bucket B':<15} {'Max Δ':<8}")
        print(f"  {'-' * 55}")
        for rank, (tv, b1, b2, md) in enumerate(id_pairs, 1):
            print(f"  {rank:<5} {tv:<8.4f} {b1:<15} {b2:<15} {md:<8.4f}")

        # Top-3 moves for each bucket
        print("\n  === TOP-3 MOVES PER BUCKET ===")
        for bucket, probs in results.items():
            top3 = sorted(
                [(probs[i], legal_info[i]["uci"]) for i in range(len(probs))], reverse=True
            )[:3]
            print(f"  {bucket:>12}: {[(f'{p:.3f}', u) for p, u in top3]}")

    print("\nDone.")


def _fen_from_moves(moves: list[str]) -> str:
    import bulletchess as bc

    board = bc.Board()
    for uci in moves:
        try:
            move = bc.Move.from_uci(uci)
            if uci not in {m.uci() for m in board.legal_moves()}:
                return board.fen()
            board.apply(move)
        except Exception:
            return board.fen()
    return board.fen()


if __name__ == "__main__":
    main()
