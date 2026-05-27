"""Game-level tokenization: validation, vocab building, and per-shard processing."""

from __future__ import annotations

from pathlib import Path

import bulletchess
import polars as pl
from loguru import logger

from krasnal.sampling import sample_bool
from krasnal.time_conditioning import uniform_clock_pair
from krasnal.tokens import (
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    NO_CHECK_ID,
    YES_CHECK_ID,
    get_elo_bucket,
    get_time_control_bucket,
    load_move_vocab,
    move_key_for_ply,
    move_token_id_for_ply,
    normalize_piece_type,
    result_to_token_id,
    save_move_vocab,
    whats_on_probe_labels,
)

from .config import PreprocessConfig


class InvalidClockDataError(ValueError):
    """Raised when clock arrays are missing, malformed, or misaligned with uci_moves."""


def _validated_piece_moved(
    piece_moved: object,
    moves_list: list[str],
    *,
    context: str,
) -> list[str]:
    if not isinstance(piece_moved, list):
        raise ValueError(f"{context}: piece_moved must be a list")
    if len(piece_moved) != len(moves_list):
        raise ValueError(
            f"{context}: piece_moved length {len(piece_moved)} does not match "
            f"uci_moves length {len(moves_list)}"
        )
    try:
        return [normalize_piece_type(piece) for piece in piece_moved]
    except ValueError as exc:
        raise ValueError(f"{context}: malformed piece_moved") from exc


def _compute_check_qa_probs(
    check_count: int,
    no_check_count: int,
    check_qa_prob: float,
) -> tuple[float, float]:
    if not 0.0 <= check_qa_prob <= 1.0:
        raise ValueError(f"check_qa_prob must be in [0, 1], got {check_qa_prob}")
    if no_check_count <= 0:
        return check_qa_prob, 0.0
    p_no = check_qa_prob * (check_count / no_check_count)
    return check_qa_prob, min(max(p_no, 0.0), 1.0)


def _clock_seconds(value: object, *, context: str) -> int:
    from krasnal.config import CLOCK_IGNORE_ID

    if value is None:
        raise ValueError(f"{context}: clock value is missing")
    seconds = int(value)
    if seconds < 0:
        raise ValueError(f"{context}: clock value must be non-negative, got {seconds}")
    if seconds >= CLOCK_IGNORE_ID:
        raise ValueError(f"{context}: clock value collides with ignore sentinel")
    return seconds


def _validated_clock_arrays(
    clocks_white: object,
    clocks_black: object,
    moves_list: list[str],
    *,
    time_initial: int | None,
    context: str,
) -> tuple[list[tuple[int, int]], int]:
    if clocks_white is None or clocks_black is None:
        raise InvalidClockDataError(f"{context}: clocks_white or clocks_black is missing")
    if not isinstance(clocks_white, list) or not isinstance(clocks_black, list):
        raise InvalidClockDataError(f"{context}: clocks_white and clocks_black must be lists")

    expected_white = (len(moves_list) + 1) // 2
    expected_black = len(moves_list) // 2
    if len(clocks_white) != expected_white or len(clocks_black) != expected_black:
        raise InvalidClockDataError(
            f"{context}: clock lengths white={len(clocks_white)}, black={len(clocks_black)} "
            f"do not match expected white={expected_white}, black={expected_black}"
        )

    initial = _clock_seconds(time_initial, context=f"{context}: time_initial")
    white_remaining = initial
    black_remaining = initial
    clock_pairs: list[tuple[int, int]] = []
    for ply in range(len(moves_list)):
        if ply % 2 == 0:
            white_remaining = _clock_seconds(
                clocks_white[ply // 2],
                context=f"{context}: clocks_white[{ply // 2}]",
            )
            clock_pairs.append((white_remaining, black_remaining))
        else:
            black_remaining = _clock_seconds(
                clocks_black[ply // 2],
                context=f"{context}: clocks_black[{ply // 2}]",
            )
            clock_pairs.append((black_remaining, white_remaining))
    return clock_pairs, initial


def _build_game_tokens(
    uci_moves: str,
    is_check: list[bool],
    piece_moved: list[str],
    result: str,
    white_rating: int,
    black_rating: int,
    time_initial: int | None,
    time_increment: int | None,
    cfg: PreprocessConfig,
    p_no: float,
    clocks_white: list[int] | None = None,
    clocks_black: list[int] | None = None,
    fen: str | None = None,
) -> tuple[list[int], list[int], list[int]]:
    if not uci_moves:
        return [], [], []

    if cfg.include_what_is_on_qa:
        b = bulletchess.Board.from_fen(fen) if fen else bulletchess.Board()

    moves_list = uci_moves.split()
    piece_types = _validated_piece_moved(
        piece_moved,
        moves_list,
        context=f"game {uci_moves[:80]!r}",
    )
    move_clock_pairs, time_initial_sec = _validated_clock_arrays(
        clocks_white,
        clocks_black,
        moves_list,
        time_initial=time_initial,
        context=f"game {uci_moves[:80]!r}",
    )
    start_clock = uniform_clock_pair(time_initial_sec)

    result_tokens = []
    active_clock_ids = []
    opponent_clock_ids = []

    def append_token(
        token_id: int,
        active_clock_id: int,
        opponent_clock_id: int,
    ) -> None:
        result_tokens.append(token_id)
        active_clock_ids.append(active_clock_id)
        opponent_clock_ids.append(opponent_clock_id)

    for ply, move in enumerate(moves_list):
        piece_type = piece_types[ply]
        move_id = move_token_id_for_ply(move, ply, piece_type)
        if move_id is None:
            key = move_key_for_ply(move, ply, piece_type)
            raise ValueError(f"Move key {key!r} is missing from generated move vocab")
        active_clock_id, opponent_clock_id = move_clock_pairs[ply]
        append_token(move_id, active_clock_id, opponent_clock_id)

        if cfg.include_check_qa:
            gives_check = ply < len(is_check) and bool(is_check[ply])
            if gives_check:
                if sample_bool(
                    seed=cfg.seed, game_key=uci_moves, ply=ply, probability=cfg.check_qa_prob
                ):
                    append_token(IS_CHECK_ID, active_clock_id, opponent_clock_id)
                    append_token(YES_CHECK_ID, active_clock_id, opponent_clock_id)
            elif sample_bool(seed=cfg.seed, game_key=uci_moves, ply=ply, probability=p_no):
                append_token(IS_CHECK_ID, active_clock_id, opponent_clock_id)
                append_token(NO_CHECK_ID, active_clock_id, opponent_clock_id)

        if cfg.include_what_is_on_qa:
            parsed_move = bulletchess.Move.from_uci(move)
            try:
                b.apply(parsed_move)
            except ValueError as exc:
                raise ValueError(
                    f"game {uci_moves[:80]!r}: illegal move at ply {ply}: {move}"
                ) from exc

            if sample_bool(
                seed=cfg.seed + 20, game_key=uci_moves, ply=ply, probability=cfg.what_is_on_prob
            ):
                _, whats_on_token_id, ans_id = whats_on_probe_labels(
                    b,
                    post_move_fen=b.fen(),
                    game_key=uci_moves,
                    ply=ply,
                    seed=cfg.seed,
                )
                append_token(whats_on_token_id, active_clock_id, opponent_clock_id)
                append_token(ans_id, active_clock_id, opponent_clock_id)

    white_elo = get_elo_bucket(white_rating)
    black_elo = get_elo_bucket(black_rating)

    prefix_tokens = [
        GAME_START_ID,
    ]
    if cfg.time_control_enabled:
        prefix_tokens.append(get_time_control_bucket(time_initial, time_increment))
    prefix_tokens.extend([result_to_token_id(result), white_elo, black_elo])
    prefix_active, prefix_opponent = zip(*[start_clock] * len(prefix_tokens), strict=True)
    end_active, end_opponent = move_clock_pairs[-1] if move_clock_pairs else start_clock
    token_ids = prefix_tokens + result_tokens + [GAME_END_ID]
    active_ids = list(prefix_active) + active_clock_ids + [end_active]
    opponent_ids = list(prefix_opponent) + opponent_clock_ids + [end_opponent]
    if not (len(token_ids) == len(active_ids) == len(opponent_ids)):
        raise RuntimeError("Clock/token alignment failed")
    return token_ids, active_ids, opponent_ids


def process_file_streaming(
    parquet_path: Path,
    output_path: Path,
    cfg: PreprocessConfig,
) -> tuple[int, int]:
    lf = pl.scan_parquet(parquet_path)

    if cfg.include_check_qa:
        count_stats = (
            lf.select(
                pl.col("is_check")
                .list.eval(pl.element().cast(pl.UInt32), parallel=True)
                .list.sum()
                .sum()
                .alias("check_count"),
                pl.col("is_check").list.len().sum().alias("ply_count"),
            )
            .collect()
            .row(0)
        )
        check_count = int(count_stats[0] or 0)
        ply_count = int(count_stats[1] or 0)
        no_check_count = max(0, ply_count - check_count)
        _, p_no = _compute_check_qa_probs(check_count, no_check_count, cfg.check_qa_prob)
    else:
        p_no = 1.0

    invalid_clock_skips = 0

    def build_tokens_batch(batch: pl.DataFrame) -> pl.DataFrame:
        nonlocal invalid_clock_skips
        token_ids_list = []
        active_clock_ids_list = []
        opponent_clock_ids_list = []

        uci_moves_list = batch.get_column("uci_moves").to_list()
        is_check_list = batch.get_column("is_check").to_list()
        piece_moved_list = batch.get_column("piece_moved").to_list()
        result_list = batch.get_column("result").to_list()
        white_rating_list = batch.get_column("white_rating").to_list()
        black_rating_list = batch.get_column("black_rating").to_list()
        clocks_white_list = batch.get_column("clocks_white").to_list()
        clocks_black_list = batch.get_column("clocks_black").to_list()
        if cfg.time_control_enabled:
            time_initial_list = batch.get_column("time_initial").to_list()
            time_increment_list = batch.get_column("time_increment").to_list()
        else:
            time_initial_list = [None] * len(batch)
            time_increment_list = [None] * len(batch)

        has_fen = "fen" in batch.columns
        fen_list = batch.get_column("fen").to_list() if has_fen else [None] * len(batch)

        for (
            uci_moves,
            is_check,
            piece_moved,
            result,
            white_rating,
            black_rating,
            clocks_white,
            clocks_black,
            time_initial,
            time_increment,
            fen,
        ) in zip(
            uci_moves_list,
            is_check_list,
            piece_moved_list,
            result_list,
            white_rating_list,
            black_rating_list,
            clocks_white_list,
            clocks_black_list,
            time_initial_list,
            time_increment_list,
            fen_list,
            strict=True,
        ):
            try:
                token_ids, active_clock_ids, opponent_clock_ids = _build_game_tokens(
                    uci_moves=uci_moves,
                    is_check=is_check,
                    piece_moved=piece_moved,
                    result=result,
                    white_rating=white_rating,
                    black_rating=black_rating,
                    time_initial=time_initial,
                    time_increment=time_increment,
                    cfg=cfg,
                    p_no=p_no,
                    clocks_white=clocks_white,
                    clocks_black=clocks_black,
                    fen=fen,
                )
            except InvalidClockDataError:
                invalid_clock_skips += 1
                continue

            token_ids_list.append(token_ids)
            active_clock_ids_list.append(active_clock_ids)
            opponent_clock_ids_list.append(opponent_clock_ids)

        if not token_ids_list:
            return pl.DataFrame(
                schema={
                    "token_ids": pl.List(pl.UInt16),
                    "active_clock_ids": pl.List(pl.UInt32),
                    "opponent_clock_ids": pl.List(pl.UInt32),
                }
            )

        return pl.DataFrame(
            {
                "token_ids": token_ids_list,
                "active_clock_ids": active_clock_ids_list,
                "opponent_clock_ids": opponent_clock_ids_list,
            },
            schema={
                "token_ids": pl.List(pl.UInt16),
                "active_clock_ids": pl.List(pl.UInt32),
                "opponent_clock_ids": pl.List(pl.UInt32),
            },
        )

    row_count = lf.select(pl.len()).collect().item()
    lf.map_batches(
        build_tokens_batch,
        schema={
            "token_ids": pl.List(pl.UInt16),
            "active_clock_ids": pl.List(pl.UInt32),
            "opponent_clock_ids": pl.List(pl.UInt32),
        },
    ).sink_parquet(output_path)
    return row_count, invalid_clock_skips


def build_move_vocab_from_corpus(
    parquet_files: list[Path],
    *,
    piece_aware_moves: bool,
    side_prefixed_moves: bool,
    output_path: Path,
) -> dict:
    move_keys: set[str] = set()
    total_plies = 0

    for parquet_path in parquet_files:
        schema = pl.scan_parquet(parquet_path).collect_schema()
        missing_columns = {"uci_moves", "piece_moved"} - set(schema.names())
        if missing_columns:
            raise ValueError(
                f"{parquet_path} is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        df = pl.read_parquet(parquet_path, columns=["uci_moves", "piece_moved"])
        uci_moves_col = df.get_column("uci_moves").to_list()
        piece_moved_col = df.get_column("piece_moved").to_list()

        for row_idx, (uci_moves, piece_moved) in enumerate(
            zip(uci_moves_col, piece_moved_col, strict=True)
        ):
            if not uci_moves or not piece_moved:
                continue
            moves_list = uci_moves.split()
            piece_types = _validated_piece_moved(
                piece_moved,
                moves_list,
                context=f"{parquet_path.name} row {row_idx}",
            )

            total_plies += len(moves_list)
            for ply, (move, piece_type) in enumerate(zip(moves_list, piece_types, strict=True)):
                key = move
                if piece_aware_moves:
                    key = f"{piece_type}:{key}"
                if side_prefixed_moves:
                    side = "w:" if ply % 2 == 0 else "b:"
                    key = f"{side}{key}"
                move_keys.add(key)

    artifact = save_move_vocab(
        output_path,
        move_keys,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )
    logger.info(
        "Wrote {} move keys from {} plies to {}",
        len(move_keys),
        total_plies,
        output_path,
    )
    return artifact


def process_one_shard(
    parquet_path: Path,
    output_path: Path,
    cfg: PreprocessConfig,
) -> tuple[str, int, int, str, dict[str, int]]:
    """Multiprocess worker: load vocab, tokenize one parquet shard, compute stats."""
    from .stats import _token_mix_raw_sums

    logger.info("Started processing {} -> {}", parquet_path.name, output_path.name)
    load_move_vocab(
        cfg.move_vocab_path,
        piece_aware_moves=cfg.piece_aware_moves,
        side_prefixed_moves=cfg.side_prefixed_moves,
    )
    count, invalid_clock_skips = process_file_streaming(parquet_path, output_path, cfg)
    raw_sums = _token_mix_raw_sums(pl.scan_parquet(output_path).select("token_ids"))
    return parquet_path.name, count, invalid_clock_skips, output_path.name, raw_sums
