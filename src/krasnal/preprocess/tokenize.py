"""Game-level tokenization: validation, vocab building, and per-shard processing."""

from __future__ import annotations

from collections import Counter
from hashlib import blake2b
from pathlib import Path

import bulletchess
import polars as pl
import pyarrow.parquet as pq
from loguru import logger

from krasnal.preprocess.pack import PackedWindowBuilder
from krasnal.time_conditioning import uniform_clock_pair
from krasnal.tokens import (
    BLACK_PREFIX,
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    MAX_SIDE_MATERIAL,
    MOVE_TO_ID,
    NO_CHECK_ID,
    OPP_MATERIAL_START_ID,
    PIECE_MATERIAL_VALUES,
    WHITE_PREFIX,
    YES_CHECK_ID,
    get_elo_bucket,
    get_time_control_bucket,
    load_move_vocab,
    move_key_for_ply,
    normalize_piece_type,
    result_to_token_id,
    save_move_vocab,
    whats_on_probe_labels,
)

from .config import PreprocessConfig

_TOKENIZED_SCHEMA = {
    "token_ids": pl.List(pl.UInt16),
    "active_clock_ids": pl.List(pl.UInt32),
    "opponent_clock_ids": pl.List(pl.UInt32),
}

_RAW_COLUMNS = [
    "uci_moves",
    "is_check",
    "piece_moved",
    "result",
    "white_rating",
    "black_rating",
    "clocks_white",
    "clocks_black",
    "time_initial",
    "time_increment",
]

_TokenizedRows = tuple[list[list[int]], list[list[int]], list[list[int]]]
_MATERIAL_VALUE_BY_PIECE_TYPE = {
    bulletchess.PAWN: PIECE_MATERIAL_VALUES["pawn"],
    bulletchess.KNIGHT: PIECE_MATERIAL_VALUES["knight"],
    bulletchess.BISHOP: PIECE_MATERIAL_VALUES["bishop"],
    bulletchess.ROOK: PIECE_MATERIAL_VALUES["rook"],
    bulletchess.QUEEN: PIECE_MATERIAL_VALUES["queen"],
    bulletchess.KING: PIECE_MATERIAL_VALUES["king"],
}


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


def _sample_bool_with_prefix(prefix: bytes, ply: int, probability: float) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    digest = blake2b(prefix + str(ply).encode(), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big") / 2**64
    return value < probability


def _move_token_id_for_ply(
    move: str,
    ply: int,
    piece_type: str,
    cfg: PreprocessConfig,
) -> int | None:
    key = move
    if cfg.piece_aware_moves:
        key = f"{piece_type}:{key}"
    if cfg.side_prefixed_moves:
        key = f"{WHITE_PREFIX if ply % 2 == 0 else BLACK_PREFIX}{key}"
    return MOVE_TO_ID.get(key)


def _piece_material_value(piece: bulletchess.Piece) -> int:
    return _MATERIAL_VALUE_BY_PIECE_TYPE[piece.piece_type]


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


def _validated_clock_initial(
    clocks_white: object,
    clocks_black: object,
    moves_list: list[str],
    *,
    time_initial: int | None,
    context: str,
) -> int:
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

    return _clock_seconds(time_initial, context=f"{context}: time_initial")


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
) -> tuple[list[int], list[int], list[int]]:
    if not uci_moves:
        return [], [], []

    b = bulletchess.Board()

    moves_list = uci_moves.split()
    piece_types = _validated_piece_moved(
        piece_moved,
        moves_list,
        context=f"game {uci_moves[:80]!r}",
    )
    context = f"game {uci_moves[:80]!r}"
    time_initial_sec = _validated_clock_initial(
        clocks_white,
        clocks_black,
        moves_list,
        time_initial=time_initial,
        context=context,
    )
    start_clock = uniform_clock_pair(time_initial_sec)
    white_remaining = time_initial_sec
    black_remaining = time_initial_sec
    end_active = time_initial_sec
    end_opponent = time_initial_sec
    white_material = 39
    black_material = 39

    result_tokens = []
    active_clock_ids = []
    opponent_clock_ids = []
    check_sample_prefix = f"{cfg.seed}|{uci_moves}|".encode()
    whats_on_sample_prefix = f"{cfg.seed + 20}|{uci_moves}|".encode()

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
        move_id = _move_token_id_for_ply(move, ply, piece_type, cfg)
        if move_id is None:
            key = move_key_for_ply(
                move,
                ply,
                piece_type,
                piece_aware_moves=cfg.piece_aware_moves,
                side_prefixed_moves=cfg.side_prefixed_moves,
            )
            raise ValueError(f"Move key {key!r} is missing from generated move vocab")
        if ply % 2 == 0:
            white_remaining = _clock_seconds(
                clocks_white[ply // 2],
                context=f"{context}: clocks_white[{ply // 2}]",
            )
            active_clock_id, opponent_clock_id = white_remaining, black_remaining
        else:
            black_remaining = _clock_seconds(
                clocks_black[ply // 2],
                context=f"{context}: clocks_black[{ply // 2}]",
            )
            active_clock_id, opponent_clock_id = black_remaining, white_remaining
        end_active, end_opponent = active_clock_id, opponent_clock_id
        append_token(move_id, active_clock_id, opponent_clock_id)

        parsed_move = bulletchess.Move.from_uci(move)
        mover = b[parsed_move.origin]
        if mover is None:
            raise ValueError(f"game {uci_moves[:80]!r}: no piece at ply {ply}: {move}")
        mover_is_white = mover.color == bulletchess.WHITE
        captured = b[parsed_move.destination]
        if (
            captured is None
            and mover.piece_type == bulletchess.PAWN
            and parsed_move.origin.index() % 8 != parsed_move.destination.index() % 8
        ):
            capture_square = (
                parsed_move.destination.south()
                if mover_is_white
                else parsed_move.destination.north()
            )
            captured = b[capture_square] if capture_square is not None else None
        if captured is not None:
            if captured.color == bulletchess.WHITE:
                white_material -= _piece_material_value(captured)
            else:
                black_material -= _piece_material_value(captured)
        if parsed_move.is_promotion():
            promotion_delta = _MATERIAL_VALUE_BY_PIECE_TYPE[parsed_move.promotion] - 1
            if mover_is_white:
                white_material += promotion_delta
            else:
                black_material += promotion_delta

        try:
            b.apply(parsed_move)
        except ValueError as exc:
            raise ValueError(f"game {uci_moves[:80]!r}: illegal move at ply {ply}: {move}") from exc
        opponent_material = black_material if mover_is_white else white_material
        if opponent_material > MAX_SIDE_MATERIAL:
            raise ValueError(
                f"Opponent material {opponent_material} exceeds max tokenized value "
                f"{MAX_SIDE_MATERIAL}"
            )
        if cfg.opponent_material_enabled:
            append_token(
                OPP_MATERIAL_START_ID + opponent_material,
                active_clock_id,
                opponent_clock_id,
            )

        if cfg.include_check_qa:
            gives_check = ply < len(is_check) and bool(is_check[ply])
            if gives_check:
                if _sample_bool_with_prefix(check_sample_prefix, ply, cfg.check_qa_prob):
                    append_token(IS_CHECK_ID, active_clock_id, opponent_clock_id)
                    append_token(YES_CHECK_ID, active_clock_id, opponent_clock_id)
            elif _sample_bool_with_prefix(check_sample_prefix, ply, p_no):
                append_token(IS_CHECK_ID, active_clock_id, opponent_clock_id)
                append_token(NO_CHECK_ID, active_clock_id, opponent_clock_id)

        if cfg.include_what_is_on_qa and _sample_bool_with_prefix(
            whats_on_sample_prefix,
            ply,
            cfg.what_is_on_prob,
        ):
            _, whats_on_token_id, ans_id = whats_on_probe_labels(
                b,
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
    if cfg.outcome_conditioning_enabled:
        prefix_tokens.append(result_to_token_id(result))
    if cfg.include_elo:
        prefix_tokens.extend([white_elo, black_elo])
    prefix_active, prefix_opponent = zip(*[start_clock] * len(prefix_tokens), strict=True)
    token_ids = prefix_tokens + result_tokens + [GAME_END_ID]
    active_ids = list(prefix_active) + active_clock_ids + [end_active]
    opponent_ids = list(prefix_opponent) + opponent_clock_ids + [end_opponent]
    if not (len(token_ids) == len(active_ids) == len(opponent_ids)):
        raise RuntimeError("Clock/token alignment failed")
    return token_ids, active_ids, opponent_ids


def _check_no_probability(parquet_path: Path, cfg: PreprocessConfig) -> float:
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
    return p_no


def _tokenize_batch(
    batch: pl.DataFrame,
    cfg: PreprocessConfig,
    p_no: float,
) -> tuple[_TokenizedRows, int]:
    token_ids_list = []
    active_clock_ids_list = []
    opponent_clock_ids_list = []
    invalid_clock_skips = 0

    uci_moves_list = batch.get_column("uci_moves").to_list()
    is_check_list = batch.get_column("is_check").to_list()
    piece_moved_list = batch.get_column("piece_moved").to_list()
    result_list = batch.get_column("result").to_list()
    white_rating_list = batch.get_column("white_rating").to_list()
    black_rating_list = batch.get_column("black_rating").to_list()
    clocks_white_list = batch.get_column("clocks_white").to_list()
    clocks_black_list = batch.get_column("clocks_black").to_list()
    time_initial_list = batch.get_column("time_initial").to_list()
    time_increment_list = batch.get_column("time_increment").to_list()
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
            )
        except InvalidClockDataError:
            invalid_clock_skips += 1
            continue

        token_ids_list.append(token_ids)
        active_clock_ids_list.append(active_clock_ids)
        opponent_clock_ids_list.append(opponent_clock_ids)

    return (
        (token_ids_list, active_clock_ids_list, opponent_clock_ids_list),
        invalid_clock_skips,
    )


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
    *,
    is_eval: bool,
    pack_flush_windows: int,
    batch_size: int,
) -> tuple[str, int, int, str, dict[str, int], dict[int, int], int]:
    """Multiprocess worker: tokenize one shard and write eval Parquet or packed train data."""
    from .stats import token_mix_raw_from_counts

    logger.info("Started processing {} -> {}", parquet_path.name, output_path.name)
    load_move_vocab(
        cfg.move_vocab_path,
        piece_aware_moves=cfg.piece_aware_moves,
        side_prefixed_moves=cfg.side_prefixed_moves,
    )

    row_count = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    p_no = _check_no_probability(parquet_path, cfg)
    invalid_clock_skips = 0
    id_counts: Counter[int] = Counter()
    length_counts: Counter[int] = Counter()

    eval_batches = []
    packed_parts_dir = output_path.parent / f"{output_path.name}_parts"
    if is_eval:
        builder = None
    else:
        builder = PackedWindowBuilder(cfg.block_size, flush_every=pack_flush_windows)

    for batch in pq.ParquetFile(parquet_path).iter_batches(
        batch_size=batch_size,
        columns=_RAW_COLUMNS,
    ):
        (token_rows, active_rows, opponent_rows), skipped = _tokenize_batch(
            pl.from_arrow(batch),
            cfg,
            p_no,
        )
        invalid_clock_skips += skipped
        if not token_rows:
            continue

        id_counts.update(tid for row in token_rows for tid in row)
        length_counts.update(len(row) for row in token_rows)

        if is_eval:
            window_size = cfg.block_size + 1
            eval_batches.append(
                pl.DataFrame(
                    {
                        "token_ids": [row[:window_size] for row in token_rows],
                        "active_clock_ids": [row[:window_size] for row in active_rows],
                        "opponent_clock_ids": [row[:window_size] for row in opponent_rows],
                    },
                    schema=_TOKENIZED_SCHEMA,
                )
            )
            continue

        assert builder is not None
        keep = [idx for idx, row in enumerate(token_rows) if len(row) <= cfg.block_size]
        if keep:
            builder.feed_from_columns(
                [token_rows[idx] for idx in keep],
                [active_rows[idx] for idx in keep],
                [opponent_rows[idx] for idx in keep],
            )
            builder.drain(packed_parts_dir)
            builder.maybe_flush(packed_parts_dir)

    if is_eval:
        if eval_batches:
            pl.concat(eval_batches).write_parquet(output_path)
        else:
            pl.DataFrame(schema=_TOKENIZED_SCHEMA).write_parquet(output_path)
        rows = len(pl.read_parquet(output_path))
    else:
        assert builder is not None
        builder.finish(output_path, part_dir=packed_parts_dir)
        with (output_path / "metadata.json").open() as f:
            import json

            rows = int(json.load(f)["rows"])
        if packed_parts_dir.exists():
            import shutil

            shutil.rmtree(packed_parts_dir)

    return (
        parquet_path.name,
        row_count,
        invalid_clock_skips,
        str(output_path),
        token_mix_raw_from_counts(dict(id_counts)),
        dict(length_counts),
        rows,
    )
