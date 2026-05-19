#!/usr/bin/env -S uv run python
"""Extract move probability distributions under different conditioning tokens.

Saves a JSON file with the probability over legal moves for each
conditioning variant (varying outcome and Elo tokens) at several
test positions.

Usage::

    cd krasnal && uv run python scripts/diagnostics/conditioning_impact.py \\
        --artifact-dir artifacts/pretrain/20260515_163216 \\
        --output ../bachelor-thesis-paper/data/conditioning_impact.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from krasnal.config import CLOCK_IGNORE_ID, MOVE_VOCAB_PATH
from krasnal.inference import Game, StatelessBatchInferenceSession, load_model
from krasnal.tokens import (
    BLACK_WON_ID,
    DRAW_ID,
    ELO_1500_1599_ID,
    ELO_1800_1899_ID,
    ELO_2000_2099_ID,
    ELO_ABOVE_2200_ID,
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

# ---------------------------------------------------------------------------
# Test positions: (label, description, moves_uci)
# Each is reached by feeding UCI strings from the starting position.
# ---------------------------------------------------------------------------
TEST_POSITIONS = [
    {
        "label": "scholars_mate",
        "description": "Scholar's Mate trap — after 1.e4 e5 2.Bc4 Nc6 3.Qh5",
        "moves": ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"],
    },
    {
        "label": "italian_two_knights",
        "description": "Italian Game, Two Knights Defense — after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
    },
    {
        "label": "fried_liver",
        "description": "Fried Liver Attack — after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5"],
    },
    {
        "label": "kings_gambit",
        "description": (
            "King's Gambit Accepted, Kieseritzky — after 1.e4 e5 2.f4 exf4 3.Nf3 g5 4.h4 g4 5.Ng5"
        ),
        "moves": ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5", "h2h4", "g5g4", "f3g5"],
    },
    {
        "label": "alekhine_four_pawns",
        "description": (
            "Alekhine's Defense, Four Pawns Attack — after "
            "1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.c4 Nb6 5.f4 dxe5 6.fxe5"
        ),
        "moves": [
            "e2e4",
            "g8f6",
            "e4e5",
            "f6d5",
            "d2d4",
            "d7d6",
            "c2c4",
            "d5b6",
            "f2f4",
            "d6e5",
            "f4e5",
        ],
    },
]

# Conditioning variants
OUTCOME_IDS = [WHITE_WON_ID, DRAW_ID, BLACK_WON_ID]
ELO_PAIRS = [
    ("low (1500-1599)", ELO_1500_1599_ID, ELO_1500_1599_ID),
    ("medium (1800)", ELO_1800_1899_ID, ELO_1800_1899_ID),
    ("high (2000)", ELO_2000_2099_ID, ELO_2000_2099_ID),
    ("top (2200+)", ELO_ABOVE_2200_ID, ELO_ABOVE_2200_ID),
]


def _token_label(tid: int) -> str:
    return ID_TO_MOVE.get(tid, f"id:{tid}")


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


def _make_clock_seq(n: int) -> list[int]:
    return [CLOCK_IGNORE_ID] * n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract conditioning counterfactual probabilities"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help=(
            "Path to pretrain artifact directory (contains config.json, model.pt, move_vocab.json)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("conditioning_impact.json"),
        help="Output JSON path (default: conditioning_impact.json)",
    )
    args = parser.parse_args()

    # ---- setup ----
    artifact_dir = args.artifact_dir
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
    if args.device != "auto":
        device = torch.device(args.device)
    print(f"Device: {device}")
    model = load_model(str(artifact_dir / "model.pt"), device, gpt_config)
    print(f"Model loaded: {gpt_config.n_layer}L, {gpt_config.n_head}H, {gpt_config.n_embd}E")

    use_time = gpt_config.use_time_conditioning
    session = StatelessBatchInferenceSession(model, device)

    # ---- build output ----
    output = {"positions": []}

    for pos in TEST_POSITIONS:
        label = pos["label"]
        moves = pos["moves"]
        fen = _fen_from_moves(moves)
        print(f"\n=== {label} ({pos['description']}) ===")
        print(f"  FEN: {fen}")

        # Build a reference board to find legal moves
        import bulletchess as bc

        ref_board = bc.Board()
        for uci in moves:
            ref_board.apply(bc.Move.from_uci(uci))
        legal = legal_token_ids(ref_board)
        legal_info = [
            {"token_id": tid, "token_label": _token_label(tid), "uci": to_uci(tid) or ""}
            for tid in sorted(legal)
        ]
        print(f"  Legal moves: {len(legal)}")

        # Build all variant games
        games: list[Game] = []
        variant_info: list[dict] = []

        for outcome_id in OUTCOME_IDS:
            for elo_label, white_elo, black_elo in ELO_PAIRS:
                game = Game(
                    white_elo_token=white_elo,
                    black_elo_token=black_elo,
                    time_control_token=TC_RAPID_INC_ID,
                    target_outcome_token=outcome_id,
                )
                for uci in moves:
                    game.feed_uci(uci)
                games.append(game)
                variant_info.append(
                    {
                        "outcome_id": outcome_id,
                        "outcome_label": _token_label(outcome_id),
                        "elo_label": elo_label,
                        "white_elo_id": white_elo,
                        "white_elo_label": _token_label(white_elo),
                        "black_elo_id": black_elo,
                        "black_elo_label": _token_label(black_elo),
                    }
                )

        # Build clock sequences for all games
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

        # Organize variants
        variants_out = []
        for idx, info in enumerate(variant_info):
            probs = all_probs[idx]
            probs_list: list[float] = []
            for lm in legal_info:
                tid = lm["token_id"]
                p = float(probs[tid].item()) if tid < len(probs) else 0.0
                probs_list.append(p)
            variants_out.append({**info, "probs": probs_list})

        output["positions"].append(
            {
                "label": label,
                "description": pos["description"],
                "fen": fen,
                "moves_uci": moves,
                "legal_moves": legal_info,
                "num_legal": len(legal),
                "variants": variants_out,
            }
        )

    # ---- write output ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
