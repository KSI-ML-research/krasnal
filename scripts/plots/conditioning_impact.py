#!/usr/bin/env -S uv run python
"""Extract conditioning counterfactual probabilities for paper plots."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import bulletchess
import torch

from krasnal.config import ARTIFACTS_DIR, CLOCK_IGNORE_ID, MOVE_VOCAB_PATH
from krasnal.inference import Game, StatelessBatchInferenceSession, load_model
from krasnal.tokens import (
    BLACK_WON_ID,
    DRAW_ID,
    ELO_1500_1599_ID,
    ELO_1800_1899_ID,
    ELO_2000_2099_ID,
    ELO_ABOVE_2200_ID,
    ID_TO_MOVE,
    TC_BLITZ_NO_INC_ID,
    TC_CLASSICAL_ID,
    TC_RAPID_INC_ID,
    TC_RAPID_NO_INC_ID,
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

ELO_POSITION = {
    "label": "italian_two_knights",
    "description": "Italian Game, Two Knights Defense after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6",
    "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
}
RESULT_POSITION = {
    "label": "scholars_mate",
    "description": "Scholar's Mate trap after 1.e4 e5 2.Bc4 Nc6 3.Qh5",
    "moves": ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"],
}
TIME_POSITIONS = [ELO_POSITION]

ELO_VARIANTS = (
    ("low (1500-1599)", ELO_1500_1599_ID, ELO_1500_1599_ID),
    ("medium (1800)", ELO_1800_1899_ID, ELO_1800_1899_ID),
    ("high (2000)", ELO_2000_2099_ID, ELO_2000_2099_ID),
    ("top (2200+)", ELO_ABOVE_2200_ID, ELO_ABOVE_2200_ID),
)
RESULT_VARIANTS = (
    ("<white_won>", WHITE_WON_ID),
    ("<draw>", DRAW_ID),
    ("<black_won>", BLACK_WON_ID),
)
TIME_VARIANTS = (
    ("blitz no increment", TC_BLITZ_NO_INC_ID),
    ("rapid no increment", TC_RAPID_NO_INC_ID),
    ("rapid increment", TC_RAPID_INC_ID),
    ("classical", TC_CLASSICAL_ID),
)


def _resolve_artifact_dir(path: Path) -> Path:
    if path.name != "LATEST":
        return path
    candidates = sorted(
        (ARTIFACTS_DIR / "pretrain").glob("*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.pt").is_file():
            return candidate
    raise FileNotFoundError(f"No pretrain artifact found in {ARTIFACTS_DIR / 'pretrain'}")


def _token_label(token_id: int) -> str:
    return ID_TO_MOVE.get(token_id, f"id:{token_id}")


def _board_from_moves(moves: list[str]) -> bulletchess.Board:
    board = bulletchess.Board()
    for uci in moves:
        move = bulletchess.Move.from_uci(uci)
        if uci not in {m.uci() for m in board.legal_moves()}:
            raise ValueError(f"Illegal diagnostic move {uci!r} in line {moves}")
        board.apply(move)
    return board


def _clock_sequences(games: list[Game]) -> list[list[int]]:
    return [[CLOCK_IGNORE_ID] * len(game.context_tokens()) for game in games]


def _legal_info(board: bulletchess.Board) -> list[dict]:
    return [
        {"token_id": token_id, "token_label": _token_label(token_id), "uci": to_uci(token_id) or ""}
        for token_id in sorted(legal_token_ids(board))
    ]


def _infer_variants(
    *,
    session: StatelessBatchInferenceSession,
    use_time: bool,
    games: list[Game],
    variant_info: list[dict],
    legal_info: list[dict],
) -> list[dict]:
    if use_time:
        logits = session.get_legal_logits_batch(
            games,
            active_clock_sequences=_clock_sequences(games),
            opponent_clock_sequences=_clock_sequences(games),
        )
    else:
        logits = session.get_legal_logits_batch(games)
    probs = torch.softmax(logits, dim=-1)
    return [
        {
            **info,
            "probs": [float(probs[idx, item["token_id"]].item()) for item in legal_info],
        }
        for idx, info in enumerate(variant_info)
    ]


def _position_payload(position: dict, legal_info: list[dict], variants: list[dict]) -> dict:
    board = _board_from_moves(position["moves"])
    return {
        "label": position["label"],
        "description": position["description"],
        "fen": board.fen(),
        "moves_uci": position["moves"],
        "legal_moves": legal_info,
        "num_legal": len(legal_info),
        "variants": variants,
    }


def _elo_payload(session: StatelessBatchInferenceSession, use_time: bool) -> dict:
    board = _board_from_moves(ELO_POSITION["moves"])
    legal_info = _legal_info(board)
    games: list[Game] = []
    variant_info: list[dict] = []
    for elo_label, white_elo, black_elo in ELO_VARIANTS:
        game = _game(ELO_POSITION["moves"], WHITE_WON_ID, white_elo, black_elo, TC_RAPID_INC_ID)
        games.append(game)
        variant_info.append(
            _variant_info(
                outcome_id=WHITE_WON_ID,
                elo_label=elo_label,
                white_elo=white_elo,
                black_elo=black_elo,
                time_control=TC_RAPID_INC_ID,
                time_control_label="rapid increment",
            )
        )
    variants = _infer_variants(
        session=session,
        use_time=use_time,
        games=games,
        variant_info=variant_info,
        legal_info=legal_info,
    )
    return _position_payload(ELO_POSITION, legal_info, variants)


def _result_payload(session: StatelessBatchInferenceSession, use_time: bool) -> dict:
    board = _board_from_moves(RESULT_POSITION["moves"])
    legal_info = _legal_info(board)
    games: list[Game] = []
    variant_info: list[dict] = []
    for outcome_label, outcome_id in RESULT_VARIANTS:
        game = _game(
            RESULT_POSITION["moves"],
            outcome_id,
            ELO_2000_2099_ID,
            ELO_2000_2099_ID,
            TC_RAPID_INC_ID,
        )
        games.append(game)
        variant_info.append(
            _variant_info(
                outcome_id=outcome_id,
                outcome_label=outcome_label,
                elo_label="high (2000)",
                white_elo=ELO_2000_2099_ID,
                black_elo=ELO_2000_2099_ID,
                time_control=TC_RAPID_INC_ID,
                time_control_label="rapid increment",
            )
        )
    variants = _infer_variants(
        session=session,
        use_time=use_time,
        games=games,
        variant_info=variant_info,
        legal_info=legal_info,
    )
    return _position_payload(RESULT_POSITION, legal_info, variants)


def _time_payloads(session: StatelessBatchInferenceSession, use_time: bool) -> list[dict]:
    payloads = []
    for position in TIME_POSITIONS:
        board = _board_from_moves(position["moves"])
        legal_info = _legal_info(board)
        games: list[Game] = []
        variant_info: list[dict] = []
        for time_label, time_id in TIME_VARIANTS:
            game = _game(
                position["moves"],
                WHITE_WON_ID,
                ELO_2000_2099_ID,
                ELO_2000_2099_ID,
                time_id,
            )
            games.append(game)
            variant_info.append(
                _variant_info(
                    outcome_id=WHITE_WON_ID,
                    elo_label="high (2000)",
                    white_elo=ELO_2000_2099_ID,
                    black_elo=ELO_2000_2099_ID,
                    time_control=time_id,
                    time_control_label=time_label,
                )
            )
        variants = _infer_variants(
            session=session,
            use_time=use_time,
            games=games,
            variant_info=variant_info,
            legal_info=legal_info,
        )
        payloads.append(_position_payload(position, legal_info, variants))
    return payloads


def _game(
    moves: list[str],
    outcome_id: int,
    white_elo: int,
    black_elo: int,
    time_control: int,
) -> Game:
    game = Game(
        white_elo_token=white_elo,
        black_elo_token=black_elo,
        time_control_token=time_control,
        target_outcome_token=outcome_id,
    )
    for uci in moves:
        game.feed_uci(uci)
    return game


def _variant_info(
    *,
    outcome_id: int,
    elo_label: str,
    white_elo: int,
    black_elo: int,
    time_control: int,
    time_control_label: str,
    outcome_label: str | None = None,
) -> dict:
    return {
        "outcome_id": outcome_id,
        "outcome_label": outcome_label or _token_label(outcome_id),
        "elo_label": elo_label,
        "white_elo_id": white_elo,
        "white_elo_label": _token_label(white_elo),
        "black_elo_id": black_elo,
        "black_elo_label": _token_label(black_elo),
        "time_control_id": time_control,
        "time_control_label": time_control_label,
        "time_control_token_label": _token_label(time_control),
    }


def _max_tv(variants: list[dict]) -> tuple[float, str, str]:
    best = (0.0, "", "")
    for left, right in combinations(variants, 2):
        tv = 0.5 * sum(abs(a - b) for a, b in zip(left["probs"], right["probs"], strict=True))
        if tv > best[0]:
            best = (tv, left["time_control_label"], right["time_control_label"])
    return best


def _top_move(variant: dict, legal_info: list[dict]) -> tuple[int, str, float]:
    idx = max(range(len(variant["probs"])), key=lambda i: variant["probs"][i])
    return idx, legal_info[idx]["uci"], variant["probs"][idx]


def _max_top1_change(position: dict) -> tuple[float, bool, str, str, str, str, float, float]:
    best = (0.0, False, "", "", "", "", 0.0, 0.0)
    legal_info = position["legal_moves"]
    for left, right in combinations(position["variants"], 2):
        left_idx, left_move, left_prob = _top_move(left, legal_info)
        right_idx, right_move, right_prob = _top_move(right, legal_info)
        changed = left_idx != right_idx
        score = abs(left_prob - right_prob)
        if (changed and not best[1]) or (changed == best[1] and score > best[0]):
            best = (
                score,
                changed,
                left["time_control_label"],
                right["time_control_label"],
                left_move,
                right_move,
                left_prob,
                right_prob,
            )
    return best


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("conditioning_impact.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact_dir = _resolve_artifact_dir(args.artifact_dir)
    config = read_model_config_json(artifact_dir / "config.json")
    gpt_config = gpt_config_from_artifact_dict(config)

    vocab_path = artifact_dir / "move_vocab.json"
    if not vocab_path.is_file():
        vocab_path = MOVE_VOCAB_PATH
    load_move_vocab(
        vocab_path,
        piece_aware_moves=bool(config.get("piece_aware_moves", True)),
        side_prefixed_moves=bool(config.get("side_prefixed_moves", True)),
    )

    device = resolve_runtime_device() if args.device == "auto" else torch.device(args.device)
    model = load_model(str(artifact_dir / "model.pt"), device, gpt_config)
    session = StatelessBatchInferenceSession(model, device)
    use_time = gpt_config.use_time_conditioning

    positions = [_elo_payload(session, use_time), _result_payload(session, use_time)]
    time_positions = _time_payloads(session, use_time)
    output = {"positions": [*positions, *time_positions], "time_control_candidates": time_positions}

    print(f"Artifact: {artifact_dir}")
    print(f"Device: {device}")
    print("Time-control probe max TV distances:")
    ranked = sorted(
        ((_max_tv(position["variants"]), position["label"]) for position in time_positions),
        reverse=True,
    )
    for (tv, left, right), label in ranked:
        print(f"  {label:<24} TV={tv:.4f} ({left} vs {right})")
    print("Time-control probe top-1 changes:")
    top1_ranked = sorted(
        ((_max_top1_change(position), position["label"]) for position in time_positions),
        reverse=True,
    )
    for (
        score,
        changed,
        left,
        right,
        left_move,
        right_move,
        left_prob,
        right_prob,
    ), label in top1_ranked:
        marker = "MOVE" if changed else "prob"
        print(
            f"  {label:<24} {marker:<4} {left}={left_move} ({left_prob:.1%}) "
            f"vs {right}={right_move} ({right_prob:.1%}), delta={score:.1%}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
