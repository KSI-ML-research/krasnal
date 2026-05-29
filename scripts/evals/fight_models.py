#!/usr/bin/env python
"""Run a head-to-head match between two Krasnal model artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import bulletchess
import torch

from krasnal.tokens import BLACK_WON_ID, WHITE_WON_ID
from krasnal.uci_engine.go_params import GoParams
from krasnal.uci_engine.provider import ModelProvider


@dataclass
class GameResult:
    game: int
    white: str
    black: str
    white_elo: int
    black_elo: int
    result: str
    reason: str
    plies: int
    moves: list[str]


def _artifact_dir(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    return p.parent if p.name == "model.pt" else p


def _load_move_vocab_payload(artifact_dir: Path) -> dict:
    with (artifact_dir / "move_vocab.json").open() as f:
        return json.load(f)


def _validate_compatible_move_vocabs(artifact_a: Path, artifact_b: Path) -> None:
    vocab_a = _load_move_vocab_payload(artifact_a)
    vocab_b = _load_move_vocab_payload(artifact_b)
    manifest_a = {
        key: value for key, value in vocab_a["manifest"].items() if key != "generation_timestamp"
    }
    manifest_b = {
        key: value for key, value in vocab_b["manifest"].items() if key != "generation_timestamp"
    }
    if manifest_a != manifest_b or vocab_a["vocab"] != vocab_b["vocab"]:
        raise ValueError(
            "Model artifacts use different move vocabularies. "
            "The current fight runner loads both models in one process and requires "
            "identical token ids."
        )


def _load_provider(path: str, device: torch.device | None) -> ModelProvider:
    artifact_dir = _artifact_dir(path)
    return ModelProvider.from_artifact_dir(artifact_dir, device=device)


def _configure_sampling(
    provider: ModelProvider,
    *,
    temperature: float,
    top_p: float,
) -> None:
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    provider.temperature = temperature
    provider.top_p = top_p


def _terminal_result(board: bulletchess.Board, last_side: str) -> tuple[str, str] | None:
    if board in bulletchess.CHECKMATE:
        return ("1-0" if last_side == "white" else "0-1"), "checkmate"
    if board in bulletchess.DRAW:
        return "1/2-1/2", "draw"
    if not board.legal_moves():
        return "1/2-1/2", "no legal moves"
    return None


def _play_game(
    *,
    game_idx: int,
    white_name: str,
    black_name: str,
    white: ModelProvider,
    black: ModelProvider,
    initial_seconds: int,
    increment_seconds: int,
    white_elo: int,
    black_elo: int,
    max_plies: int,
) -> GameResult:
    board = bulletchess.Board()
    moves: list[str] = []

    for provider in (white, black):
        provider.apply_setoption("krasnalInitialSeconds", str(initial_seconds))
        provider.apply_setoption("krasnalIncrementSeconds", str(increment_seconds))
    white.apply_setoption("krasnalWhiteElo", str(white_elo))
    white.apply_setoption("krasnalBlackElo", str(black_elo))
    black.apply_setoption("krasnalWhiteElo", str(white_elo))
    black.apply_setoption("krasnalBlackElo", str(black_elo))
    white.reset_session(WHITE_WON_ID)
    black.reset_session(BLACK_WON_ID)

    for ply in range(max_plies):
        side = "white" if board.turn == bulletchess.WHITE else "black"
        provider = white if side == "white" else black
        provider.set_go_params(
            GoParams(
                wtime_ms=initial_seconds * 1000,
                btime_ms=initial_seconds * 1000,
                winc_ms=increment_seconds * 1000,
                binc_ms=increment_seconds * 1000,
            )
        )
        move_uci = provider.get_best_move(" ".join(moves))
        legal = {move.uci() for move in board.legal_moves()}
        if move_uci not in legal:
            result = "0-1" if side == "white" else "1-0"
            return GameResult(
                game=game_idx,
                white=white_name,
                black=black_name,
                white_elo=white_elo,
                black_elo=black_elo,
                result=result,
                reason=f"illegal move by {side}: {move_uci}",
                plies=ply,
                moves=moves,
            )

        board.apply(bulletchess.Move.from_uci(move_uci))
        moves.append(move_uci)
        terminal = _terminal_result(board, side)
        if terminal is not None:
            result, reason = terminal
            return GameResult(
                game=game_idx,
                white=white_name,
                black=black_name,
                white_elo=white_elo,
                black_elo=black_elo,
                result=result,
                reason=reason,
                plies=len(moves),
                moves=moves,
            )

    return GameResult(
        game=game_idx,
        white=white_name,
        black=black_name,
        white_elo=white_elo,
        black_elo=black_elo,
        result="1/2-1/2",
        reason=f"max plies reached ({max_plies})",
        plies=len(moves),
        moves=moves,
    )


def _score_for(name: str, result: GameResult) -> float:
    if result.result == "1/2-1/2":
        return 0.5
    if result.result == "1-0":
        return 1.0 if result.white == name else 0.0
    return 1.0 if result.black == name else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", help="Artifact dir or model.pt for model A")
    parser.add_argument("model_b", help="Artifact dir or model.pt for model B")
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--initial-seconds", type=int, default=180)
    parser.add_argument("--increment-seconds", type=int, default=2)
    parser.add_argument("--elo-a", type=int, default=2200)
    parser.add_argument("--elo-b", type=int, default=2200)
    parser.add_argument("--device", default=None, help="cpu, cuda, mps, or auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature-a", type=float, default=None)
    parser.add_argument("--temperature-b", type=float, default=None)
    parser.add_argument("--top-p-a", type=float, default=None)
    parser.add_argument("--top-p-b", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--disable-outcome-a",
        action="store_true",
        help="Force model A to omit <white_won>/<black_won> prompt tokens.",
    )
    parser.add_argument(
        "--disable-outcome-b",
        action="store_true",
        help="Force model B to omit <white_won>/<black_won> prompt tokens.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = None if args.device in (None, "auto") else torch.device(args.device)
    artifact_a = _artifact_dir(args.model_a)
    artifact_b = _artifact_dir(args.model_b)
    _validate_compatible_move_vocabs(artifact_a, artifact_b)
    provider_a = _load_provider(str(artifact_a), device)
    provider_b = _load_provider(str(artifact_b), device)
    temperature_a = args.temperature if args.temperature_a is None else args.temperature_a
    temperature_b = args.temperature if args.temperature_b is None else args.temperature_b
    top_p_a = args.top_p if args.top_p_a is None else args.top_p_a
    top_p_b = args.top_p if args.top_p_b is None else args.top_p_b
    _configure_sampling(provider_a, temperature=temperature_a, top_p=top_p_a)
    _configure_sampling(provider_b, temperature=temperature_b, top_p=top_p_b)
    if args.disable_outcome_a:
        provider_a.outcome_conditioning_enabled = False
    if args.disable_outcome_b:
        provider_b.outcome_conditioning_enabled = False

    results: list[GameResult] = []
    for idx in range(args.games):
        if idx % 2 == 0:
            white_name, black_name = args.name_a, args.name_b
            white, black = provider_a, provider_b
            white_elo = args.elo_a
            black_elo = args.elo_b
        else:
            white_name, black_name = args.name_b, args.name_a
            white, black = provider_b, provider_a
            white_elo = args.elo_b
            black_elo = args.elo_a

        result = _play_game(
            game_idx=idx + 1,
            white_name=white_name,
            black_name=black_name,
            white=white,
            black=black,
            initial_seconds=args.initial_seconds,
            increment_seconds=args.increment_seconds,
            white_elo=white_elo,
            black_elo=black_elo,
            max_plies=args.max_plies,
        )
        results.append(result)
        print(
            f"game {result.game}: {result.white}({result.white_elo}) "
            f"vs {result.black}({result.black_elo}) "
            f"{result.result} ({result.reason}, {result.plies} plies)"
        )

    score_a = sum(_score_for(args.name_a, result) for result in results)
    score_b = sum(_score_for(args.name_b, result) for result in results)
    summary = {
        "model_a": args.name_a,
        "model_b": args.name_b,
        "sampling": {
            args.name_a: {"temperature": temperature_a, "top_p": top_p_a},
            args.name_b: {"temperature": temperature_b, "top_p": top_p_b},
            "seed": args.seed,
        },
        "elo": {
            args.name_a: args.elo_a,
            args.name_b: args.elo_b,
        },
        "outcome_conditioning": {
            args.name_a: provider_a.outcome_conditioning_enabled,
            args.name_b: provider_b.outcome_conditioning_enabled,
        },
        "score_a": score_a,
        "score_b": score_b,
        "games": [asdict(result) for result in results],
    }
    print(f"score: {args.name_a} {score_a:g} - {score_b:g} {args.name_b}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
