#!/usr/bin/env -S uv run python
"""Large outcome-conditioning grid search (fixed Elo, vary result token).

For each tabiya we still run logits with `<white_won>`, `<draw>`, and `<black_won>`
in one batch (same as inference). **Rankings and printed scores use only total
variation between `<white_won>` and `<black_won>`**—draw is omitted from separation
metrics and from single-move spreads.

    cd krasnal && uv run python scripts/diagnostics/outcome_conditioning_grid_search.py \\
        --artifact-dir artifacts/pretrain/20260515_210200 \\
        --output ../bachelor-thesis-paper/data/outcome_conditioning_grid_results.json \\
        --print-top 25

Uses only two symmetric Elo buckets (1500--1599 and 2200+), consistent with an
impact figure that contrasts learner-style play with elite play.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bulletchess as bc
import torch

from krasnal.config import CLOCK_IGNORE_ID, MOVE_VOCAB_PATH
from krasnal.inference import Game, StatelessBatchInferenceSession, load_model
from krasnal.tokens import (
    BLACK_WON_ID,
    DRAW_ID,
    ELO_1500_1599_ID,
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

# -----------------------------------------------------------------------------
# Candidate positions (UCI ply list from the initial position).
# -----------------------------------------------------------------------------
GRID_SEARCH_POSITIONS: list[dict[str, Any]] = [
    {
        "label": "scholars_mate",
        "description": "Scholar's Mate — after 1.e4 e5 2.Bc4 Nc6 3.Qh5",
        "moves": ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"],
    },
    {
        "label": "italian_two_knights",
        "description": "Italian Two Knights — after 3…Nf6",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
    },
    {
        "label": "italian_slow_g3",
        "description": "Italian quiet with g3 — after 3…Nf6 4.g3",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "g2g3"],
    },
    {
        "label": "fried_liver_ng5",
        "description": "Fried Liver — after 4.Ng5",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5"],
    },
    {
        "label": "fried_liver_ulvestad_na5",
        "description": "Two Knights Ultravsted — after 5.exd5 Na5",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5", "e4d5", "c6a5"],
    },
    {
        "label": "kings_gambit_kieseritzky",
        "description": "King's Gambit Accepted — Kieseritzky line after 5.Ng5",
        "moves": ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5", "h2h4", "g5g4", "f3g5"],
    },
    {
        "label": "alekhine_four_pawns",
        "description": "Alekhine Four Pawns — after 6.fxe5",
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
    {
        "label": "scotch_4nxd4",
        "description": "Scotch / open center — after 4…exd4 5.Nxd4",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"],
    },
    {
        "label": "ruy_morphy_bb5xa4",
        "description": "Ruy Lopez Morphy — after …a6 6.Ba4",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"],
    },
    {
        "label": "ruy_steinitz_re1",
        "description": "Ruy Lopez Closed Steinitz — after …d6 and c3",
        "moves": [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1b5",
            "g8f6",
            "e1g1",
            "f8e7",
            "f1e1",
            "d7d6",
            "c2c3",
        ],
    },
    {
        "label": "caro_kann_bf5_advanced",
        "description": "Caro-Kann Advance — Bishop on f5",
        "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5", "g1f3", "e7e6"],
    },
    {
        "label": "french_advanced",
        "description": "French Advance — symmetrical pawn spear",
        "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "b8c6", "g1f3", "d8c7"],
    },
    {
        "label": "nimzo_indian",
        "description": "Nimzo-Indian Classical — Bishop on b4",
        "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "g1f3", "b7b6", "g2g3"],
    },
    {
        "label": "qgd_classical_bg5",
        "description": "Queen's Gambit Declined — Bg5 main line",
        "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "h7h6", "g5h4"],
    },
    {
        "label": "sicilian_open_a6_nc3",
        "description": "Open Sicilian — Najdorf-style …a6 and Nc3",
        "moves": ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "a7a6", "b1c3"],
    },
    {
        "label": "sicilian_dragon_CASTLE",
        "description": "Sicilian Dragon castled kingside — Yugoslav attack prelude",
        "moves": [
            "e2e4",
            "c7c5",
            "g1f3",
            "d7d6",
            "d2d4",
            "c5d4",
            "f3d4",
            "g8f6",
            "b1c3",
            "g7g6",
            "f1e2",
            "f8g7",
            "e1g1",
            "e8g8",
            "f2f4",
        ],
    },
    {
        "label": "petroff_nf3_take",
        "description": "Petroff — after 5.Nf3",
        "moves": ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6", "e5f3"],
    },
    {
        "label": "philidor_steinitz",
        "description": "Spanish Exchange style — …a6 Bxc6 dxc6 O-O …Be7 Re1 setup",
        "moves": [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1b5",
            "a7a6",
            "b5c6",
            "d7c6",
            "e1g1",
            "f8e7",
            "f1e1",
        ],
    },
    {
        "label": "trompowsky_h6_bh4",
        "description": "Trompowsky — blocked bishop on h4",
        "moves": ["d2d4", "g8f6", "c1g5", "h7h6", "g5h4", "g7g6", "b1d2"],
    },
    {
        "label": "london_system",
        "description": "London System — pawn on c6 and …g6 fianchetto",
        "moves": ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "c7c5", "e2e3", "g7g6", "h2h3"],
    },
    {
        "label": "englund_nf3_setup",
        "description": "Englund-style counter — after dx e5 …Nc6 and Nf3",
        "moves": ["d2d4", "e7e5", "d4e5", "b8c6", "g1f3"],
    },
    {
        "label": "blackmar_diemer_piece",
        "description": "Blackmar-Diemer — after pawn recapture …Qxd5",
        "moves": ["d2d4", "d7d5", "e2e4", "d5e4", "b1c3", "g8f6", "f2f3", "e4f3", "d1f3"],
    },
    {
        "label": "scandinavian_mainline_nf3",
        "description": "Scandinavian mainline central fight",
        "moves": ["e2e4", "d7d5", "e4d5", "d8d5", "g1f3", "d5d8", "d2d4", "g8f6"],
    },
    {
        "label": "queens_indian_fianchetto",
        "description": "Queen's Indian — …b6 …Bb7 with W fianchetto",
        "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7", "f1g2"],
    },
    {
        "label": "dutch_leningrad",
        "description": "Dutch Leningrad — King's Indian style",
        "moves": ["d2d4", "f7f5", "g1f3", "g8f6", "g2g3", "g7g6", "f1g2", "f8g7"],
    },
    {
        "label": "catalan_d5",
        "description": "Catalan Closed — Bishop on g2",
        "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "g1f3", "f8e7"],
    },
    {
        "label": "slav_meran_piece",
        "description": "Slav — Accepted cxd5 structure",
        "moves": ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"],
    },
    {
        "label": "semi_slav_meranian",
        "description": "Semi-Slav — triangle with Nd7 shell",
        "moves": ["d2d4", "d7d5", "g1f3", "c7c6", "c2c4", "g8f6", "e2e3", "e7e6", "b1c3", "b8d7"],
    },
    {
        "label": "semi_slav_qc7",
        "description": "Semi-Slav — …Qc7 and e3",
        "moves": ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "e2e3", "e7e6", "b1c3", "d8c7"],
    },
    {
        "label": "gruenfeld_8piece",
        "description": "Grünfeld — fianchettoed King bishop",
        "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "g1f3", "f8g7", "b1c3", "d7d5"],
    },
    {
        "label": "english_symmetrical",
        "description": "English Symmetrical Hedgehog prelude",
        "moves": ["c2c4", "c7c5", "g1f3", "g8f6", "b1c3", "e7e6", "g2g3", "b7b6", "f1g2", "c8b7"],
    },
    {
        "label": "evans_gambit_axb4",
        "description": "Evans Gambit — Black captures on b4; White about to reclaim",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4", "c5b4"],
    },
    {
        "label": "smith_morra_dcxd4",
        "description": "Sicilian Smith-Morra gambit pawn recapture accepted",
        "moves": ["e2e4", "c7c5", "d2d4", "c5d4", "c2c3", "d4c3", "b1c3"],
    },
]


def replay_moves_strict(moves: list[str]) -> tuple[bool, int | None]:
    """Return (success, ply_index_where_illegal_if_failed)."""
    board = bc.Board()
    for ply, uci in enumerate(moves):
        if uci not in {m.uci() for m in board.legal_moves()}:
            return False, ply
        board.apply(bc.Move.from_uci(uci))
    return True, None


def filter_valid_positions() -> tuple[list[dict[str, Any]], list[str]]:
    ok: list[dict[str, Any]] = []
    bad: list[str] = []
    for pos in GRID_SEARCH_POSITIONS:
        good, ply = replay_moves_strict(pos["moves"])
        if good:
            ok.append(pos)
        else:
            illegal_move = pos["moves"][ply if ply is not None else 0]
            bad.append(f"{pos['label']}: illegal at ply {ply} ({illegal_move!r})")
    return ok, bad


OUTCOME_IDS = [WHITE_WON_ID, DRAW_ID, BLACK_WON_ID]

ELO_RUNS = [
    ("elo_1500_1599", ELO_1500_1599_ID, "<elo_1500_1599>"),
    ("elo_above_2200", ELO_ABOVE_2200_ID, "<elo_above_2200>"),
]


def _token_label(tid: int) -> str:
    return ID_TO_MOVE.get(tid, f"id:{tid}")


def pairwise_tv(probs_a: list[float], probs_b: list[float]) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(probs_a, probs_b, strict=False))


def max_tv_three_way(p_w: list[float], p_d: list[float], p_b: list[float]) -> float:
    return max(
        pairwise_tv(p_w, p_d),
        pairwise_tv(p_w, p_b),
        pairwise_tv(p_d, p_b),
    )


def tv_white_vs_black(p_w: list[float], p_b: list[float]) -> float:
    return pairwise_tv(p_w, p_b)


@dataclass
class Interpret:
    """Single-move contrast for White-win vs Black-win conditioning only."""

    uci_best: str
    spread_white_vs_black: float
    p_white_on_best: float
    p_black_on_best: float

    white_minus_black_most_positive_uci: str
    delta_white_minus_black_most_positive: float
    white_minus_black_most_negative_uci: str
    delta_white_minus_black_most_negative: float


def interpret_white_vs_black(
    legal_info: list[dict[str, Any]],
    p_white: list[float],
    p_black: list[float],
) -> Interpret:
    n = len(p_white)
    spreads = [abs(p_white[i] - p_black[i]) for i in range(n)]
    i_star = max(range(n), key=lambda k: spreads[k])
    deltas_wb = [p_white[i] - p_black[i] for i in range(n)]
    i_pos = max(range(n), key=lambda k: deltas_wb[k])
    i_neg = min(range(n), key=lambda k: deltas_wb[k])
    lm = legal_info[i_star]["uci"]

    return Interpret(
        uci_best=lm,
        spread_white_vs_black=float(spreads[i_star]),
        p_white_on_best=float(p_white[i_star]),
        p_black_on_best=float(p_black[i_star]),
        white_minus_black_most_positive_uci=legal_info[i_pos]["uci"],
        delta_white_minus_black_most_positive=float(deltas_wb[i_pos]),
        white_minus_black_most_negative_uci=legal_info[i_neg]["uci"],
        delta_white_minus_black_most_negative=float(deltas_wb[i_neg]),
    )


def fen_from_moves(moves: list[str]) -> str:
    board = bc.Board()
    for uci in moves:
        board.apply(bc.Move.from_uci(uci))
    return board.fen()


def _make_clock_seq(n: int) -> list[int]:
    return [CLOCK_IGNORE_ID] * n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search outcome conditioning separation (many positions)"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outcome_conditioning_grid_results.json"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--print-top", type=int, default=20)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify UCI ladders only; skip model inference",
    )
    args = parser.parse_args()

    valid_positions, rejects = filter_valid_positions()
    if rejects:
        print("Rejected positions:")
        for r in rejects:
            print(f"  {r}")
    print(f"Using {len(valid_positions)} validated positions.")

    if args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"valid_labels": [p["label"] for p in valid_positions]}, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
        return

    config_dict = read_model_config_json(args.artifact_dir / "config.json")
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
    model = load_model(str(args.artifact_dir / "model.pt"), device, gpt_config)
    use_time = gpt_config.use_time_conditioning
    session = StatelessBatchInferenceSession(model, device)

    rows: list[dict[str, Any]] = []

    for pos in valid_positions:
        label = pos["label"]
        moves = pos["moves"]
        fen = fen_from_moves(moves)

        ref_board = bc.Board()
        for uci in moves:
            ref_board.apply(bc.Move.from_uci(uci))
        legal = legal_token_ids(ref_board)
        legal_info = [
            {"token_id": tid, "token_label": _token_label(tid), "uci": to_uci(tid) or ""}
            for tid in sorted(legal)
        ]

        for elo_slug, elo_id, elo_label_txt in ELO_RUNS:
            games: list[Game] = []
            variant_tags: list[dict[str, str]] = []

            for outcome_id in OUTCOME_IDS:
                g = Game(
                    white_elo_token=elo_id,
                    black_elo_token=elo_id,
                    time_control_token=TC_RAPID_INC_ID,
                    target_outcome_token=outcome_id,
                )
                for uci in moves:
                    g.feed_uci(uci)
                games.append(g)
                variant_tags.append(
                    {
                        "outcome_label": _token_label(outcome_id),
                    }
                )

            if use_time:
                clocks = [_make_clock_seq(len(g.context_tokens())) for g in games]
                logits_batch = session.get_legal_logits_batch(
                    games,
                    active_clock_sequences=clocks,
                    opponent_clock_sequences=clocks,
                )
            else:
                logits_batch = session.get_legal_logits_batch(games)

            probs_bt = torch.softmax(logits_batch, dim=-1)

            def probs_for_row(
                row_idx: int, probs_bt=probs_bt, legal_info=legal_info
            ) -> list[float]:
                probs = probs_bt[row_idx]
                out: list[float] = []
                for lm in legal_info:
                    tid = lm["token_id"]
                    out.append(float(probs[tid].item()) if tid < len(probs) else 0.0)
                return out

            p_w = probs_for_row(0)
            p_draw = probs_for_row(1)
            p_black = probs_for_row(2)

            tv_wb = tv_white_vs_black(p_w, p_black)
            tv_three = max_tv_three_way(p_w, p_draw, p_black)
            inter = interpret_white_vs_black(legal_info, p_w, p_black)

            rows.append(
                {
                    "position_label": label,
                    "position_description": pos["description"],
                    "fen": fen,
                    "moves_uci": moves,
                    "elo_bucket": elo_slug,
                    "elo_white_black_token": elo_label_txt,
                    "total_variation_white_vs_black": tv_wb,
                    "max_total_variation_three_outcomes": tv_three,
                    "interpretation": {
                        "widest_move_uci": inter.uci_best,
                        "p_white_on_widest_move": inter.p_white_on_best,
                        "p_black_on_widest_move": inter.p_black_on_best,
                        "widest_spread_white_vs_black": inter.spread_white_vs_black,
                        "white_minus_black_polar_moves": {
                            "more_likely_if_white_eventually_wins": {
                                "uci": inter.white_minus_black_most_positive_uci,
                                "delta_p_white_minus_black": (
                                    inter.delta_white_minus_black_most_positive
                                ),
                            },
                            "more_likely_if_black_eventually_wins": {
                                "uci": inter.white_minus_black_most_negative_uci,
                                "delta_p_white_minus_black": (
                                    inter.delta_white_minus_black_most_negative
                                ),
                            },
                        },
                    },
                }
            )

    metric = "total_variation_white_vs_black"

    def composite_key(entry: dict[str, Any]) -> tuple[float, float]:
        elo_hit = entry["elo_bucket"] == "elo_above_2200"
        return (entry[metric], elo_hit)

    rows_sorted = sorted(rows, key=composite_key, reverse=True)

    # Split leaderboards per Elo
    lb_1500 = sorted(
        [r for r in rows if r["elo_bucket"] == "elo_1500_1599"],
        key=lambda r: r[metric],
        reverse=True,
    )
    lb_2200 = sorted(
        [r for r in rows if r["elo_bucket"] == "elo_above_2200"],
        key=lambda r: r[metric],
        reverse=True,
    )

    doc = {
        "artifact_dir": str(args.artifact_dir),
        "primary_metric": metric,
        "note": (
            "Rankings use total-variation distance between distributions under "
            "<white_won> and <black_won> only. Draw-conditioning is omitted from scores."
        ),
        "counts": {"positions": len(valid_positions), "runs": len(rows)},
        "elo_buckets": ["elo_1500_1599", "elo_above_2200"],
        "leaderboard_elo_above_2200": lb_2200,
        "leaderboard_elo_1500_1599": lb_1500,
        "all_rows": rows_sorted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"\nSaved {len(rows)} evaluations to {args.output}")
    print(f"\n--- Top {args.print_top} at Elo 2200+ (TV white vs black) ---\n")
    for r in lb_2200[: args.print_top]:
        interp = r["interpretation"]
        print(
            f"{r[metric]:.4f}  {r['position_label']:<28} "
            f"{interp['widest_move_uci']}|W-B|={interp['widest_spread_white_vs_black']:.1%}"
        )
    print(f"\n--- Top {args.print_top} at Elo 1500-1599 ---\n")
    for r in lb_1500[: args.print_top]:
        interp = r["interpretation"]
        print(
            f"{r[metric]:.4f}  {r['position_label']:<28} "
            f"{interp['widest_move_uci']}|W-B|={interp['widest_spread_white_vs_black']:.1%}"
        )


if __name__ == "__main__":
    main()
