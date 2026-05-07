import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import bulletchess
import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from krasnal.config import (
    EVAL_DATASET_PATH,
    MOVE_VOCAB_PATH,
    PRETRAIN_DATASET_PATH,
    RAW_UCI_DIR,
)
from krasnal.sampling import sample_bool, whats_on_square_index
from krasnal.tokens import (
    BISHOP_ID,
    BLACK_WON_ID,
    COLORED_PIECE_TOKENS,
    DRAW_ID,
    ELO_1000_1499_ID,
    ELO_1500_1999_ID,
    ELO_2000_2499_ID,
    ELO_2500_2999_ID,
    ELO_ABOVE_3000_ID,
    ELO_BELOW_1000_ID,
    ELO_UNKNOWN_ID,
    EMPTY_ID,
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    KING_ID,
    KNIGHT_ID,
    NO_CHECK_ID,
    PAWN_ID,
    PIECE_TYPE_ALIASES,
    PIECE_TYPE_MOVED_ID,
    PIECE_TYPES,
    QUEEN_ID,
    ROOK_ID,
    SPECIAL_TOKENS,
    UNKNOWN_RESULT_ID,
    WHATS_ON_SQUARE,
    WHATS_ON_SQUARE_TOKEN_IDS,
    WHITE_WON_ID,
    YES_CHECK_ID,
    get_elo_bucket,
    load_move_vocab,
    move_key_for_ply,
    move_token_id_for_ply,
    normalize_piece_type,
    result_to_token_id,
    save_move_vocab,
)

PIECE_TYPE_TO_TOKEN_ID: dict[str, int] = {
    "pawn": PAWN_ID,
    "knight": KNIGHT_ID,
    "bishop": BISHOP_ID,
    "rook": ROOK_ID,
    "queen": QUEEN_ID,
    "king": KING_ID,
}


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


def _compute_piece_sampling_probs(
    piece_counts: dict[str, int], king_base_prob: float
) -> dict[str, float]:
    king_count = piece_counts.get("king", 0)
    if king_count <= 0:
        logger.warning("No king moves found in shard; piece Q&A sampling disabled for this shard")
        return {piece_type: 0.0 for piece_type in PIECE_TYPES}

    probs: dict[str, float] = {}
    for piece_type in PIECE_TYPES:
        count = piece_counts.get(piece_type, 0)
        if count <= 0:
            probs[piece_type] = 0.0
            continue
        probs[piece_type] = min(1.0, king_base_prob * (king_count / count))
    return probs


def _compute_piece_counts(lf: pl.LazyFrame) -> dict[str, int]:
    count_exprs = []
    for piece_type, aliases in PIECE_TYPE_ALIASES.items():
        count_exprs.append(
            pl.col("piece_moved")
            .list.eval(
                pl.element().cast(pl.Utf8).str.to_lowercase().is_in(list(aliases)).cast(pl.UInt32),
                parallel=True,
            )
            .list.sum()
            .sum()
            .alias(f"{piece_type}_count")
        )

    stats = lf.select(*count_exprs).collect().row(0)
    return {piece_type: int(stats[i] or 0) for i, piece_type in enumerate(PIECE_TYPES)}


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


def _build_game_tokens(
    uci_moves: str,
    is_check: list[bool],
    piece_moved: list[str],
    result: str,
    white_rating: int,
    black_rating: int,
    elo_bucket: int,
    include_check_qa: bool,
    check_qa_prob: float,
    normal_prob: float,
    white_unknown_prob: float,
    black_unknown_prob: float,
    both_unknown_prob: float,
    seed: int,
    p_no: float,
    include_piece_qa: bool,
    piece_sampling_probs: dict[str, float],
    fen: str | None = None,
    include_what_is_on_qa: bool = False,
    what_is_on_prob: float = 0.0,
) -> list[int]:
    if not uci_moves:
        return []

    if include_what_is_on_qa:
        b = bulletchess.Board.from_fen(fen) if fen else bulletchess.Board()

    moves_list = uci_moves.split()
    piece_types = _validated_piece_moved(
        piece_moved,
        moves_list,
        context=f"game {uci_moves[:80]!r}",
    )

    result_tokens = []
    for ply, move in enumerate(moves_list):
        piece_type = piece_types[ply]
        move_id = move_token_id_for_ply(move, ply, piece_type)
        if move_id is None:
            key = move_key_for_ply(move, ply, piece_type)
            raise ValueError(f"Move key {key!r} is missing from generated move vocab")
        result_tokens.append(move_id)

        if include_check_qa:
            gives_check = ply < len(is_check) and bool(is_check[ply])
            if gives_check:
                if sample_bool(seed=seed, game_key=uci_moves, ply=ply, probability=check_qa_prob):
                    result_tokens.extend([IS_CHECK_ID, YES_CHECK_ID])
            elif sample_bool(seed=seed, game_key=uci_moves, ply=ply, probability=p_no):
                result_tokens.extend([IS_CHECK_ID, NO_CHECK_ID])

        if include_piece_qa:
            probability = piece_sampling_probs.get(piece_type, 0.0)
            if sample_bool(seed=seed + 13, game_key=uci_moves, ply=ply, probability=probability):
                answer_token = PIECE_TYPE_TO_TOKEN_ID[piece_type]
                result_tokens.extend([PIECE_TYPE_MOVED_ID, answer_token])

        if include_what_is_on_qa:
            for m in b.legal_moves():
                if m.uci() == move:
                    b.apply(m)
                    break

            if sample_bool(
                seed=seed + 20, game_key=uci_moves, ply=ply, probability=what_is_on_prob
            ):
                post_move_fen = b.fen()
                sq_idx = whats_on_square_index(
                    post_move_fen=post_move_fen,
                    game_key=uci_moves,
                    ply=ply,
                    seed=seed,
                )
                file_char = chr(97 + (sq_idx % 8))
                rank_char = str(1 + (sq_idx // 8))
                sq_str = f"{file_char}{rank_char}"
                whats_on_token_id = WHATS_ON_SQUARE[f"<whats_on_{sq_str}>"]

                piece = b[bulletchess.Square.from_str(sq_str)]
                if piece is None:
                    ans_id = EMPTY_ID
                else:
                    color_str = "w" if str(piece.color) == "White" else "b"
                    piece_str = str(piece.piece_type).lower()
                    ans_id = COLORED_PIECE_TOKENS[f"<{color_str}:{piece_str}>"]

                result_tokens.extend([whats_on_token_id, ans_id])

    white_elo = get_elo_bucket(white_rating)
    black_elo = get_elo_bucket(black_rating)

    normal_threshold = round(normal_prob * 1000)
    white_unknown_threshold = normal_threshold + round(white_unknown_prob * 1000)
    black_unknown_threshold = white_unknown_threshold + round(black_unknown_prob * 1000)
    both_unknown_threshold = black_unknown_threshold + round(both_unknown_prob * 1000)

    if not 0 <= elo_bucket < 1000:
        raise ValueError(f"elo_bucket must be in [0, 1000), got {elo_bucket}")
    if both_unknown_threshold != 1000:
        raise ValueError(
            "Unknown ELO probabilities must sum to 1.0 in 0.001 increments; "
            f"got total={both_unknown_threshold / 1000:.3f}"
        )

    if elo_bucket < normal_threshold:
        pass
    elif elo_bucket < white_unknown_threshold:
        white_elo = ELO_UNKNOWN_ID
    elif elo_bucket < black_unknown_threshold:
        black_elo = ELO_UNKNOWN_ID
    elif elo_bucket < both_unknown_threshold:
        white_elo = ELO_UNKNOWN_ID
        black_elo = ELO_UNKNOWN_ID

    prefix_tokens = [
        GAME_START_ID,
        result_to_token_id(result),
        white_elo,
        black_elo,
    ]
    return prefix_tokens + result_tokens + [GAME_END_ID]


def process_file_streaming(
    parquet_path: Path,
    seed: int,
    output_path: Path,
    include_check_qa: bool,
    check_qa_prob: float,
    include_piece_qa: bool,
    king_base_prob: float,
    unknown_elo: dict[str, float],
    include_what_is_on_qa: bool = False,
    what_is_on_prob: float = 0.0,
) -> int:
    lf = pl.scan_parquet(parquet_path)

    if include_check_qa:
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
        _, p_no = _compute_check_qa_probs(check_count, no_check_count, check_qa_prob)
    else:
        p_no = 1.0

    piece_counts = {piece_type: 0 for piece_type in PIECE_TYPES}
    piece_sampling_probs = {piece_type: 0.0 for piece_type in PIECE_TYPES}
    if include_piece_qa:
        piece_counts = _compute_piece_counts(lf)
        piece_sampling_probs = _compute_piece_sampling_probs(piece_counts, king_base_prob)

    normal_prob = float(unknown_elo["normal_prob"])
    white_unknown_prob = float(unknown_elo["white_unknown_prob"])
    black_unknown_prob = float(unknown_elo["black_unknown_prob"])
    both_unknown_prob = float(unknown_elo["both_unknown_prob"])

    def build_tokens_batch(batch: pl.DataFrame) -> pl.DataFrame:
        token_ids_list = []

        uci_moves_list = batch.get_column("uci_moves").to_list()
        is_check_list = batch.get_column("is_check").to_list()
        piece_moved_list = batch.get_column("piece_moved").to_list()
        result_list = batch.get_column("result").to_list()
        white_rating_list = batch.get_column("white_rating").to_list()
        black_rating_list = batch.get_column("black_rating").to_list()
        elo_bucket_list = batch.get_column("elo_bucket").to_list()

        has_fen = "fen" in batch.columns
        fen_list = batch.get_column("fen").to_list() if has_fen else [None] * len(batch)

        for (
            uci_moves,
            is_check,
            piece_moved,
            result,
            white_rating,
            black_rating,
            elo_bucket,
            fen,
        ) in zip(
            uci_moves_list,
            is_check_list,
            piece_moved_list,
            result_list,
            white_rating_list,
            black_rating_list,
            elo_bucket_list,
            fen_list,
            strict=True,
        ):
            token_ids = _build_game_tokens(
                uci_moves=uci_moves,
                is_check=is_check,
                piece_moved=piece_moved,
                result=result,
                white_rating=white_rating,
                black_rating=black_rating,
                elo_bucket=elo_bucket,
                include_check_qa=include_check_qa,
                check_qa_prob=check_qa_prob,
                normal_prob=normal_prob,
                white_unknown_prob=white_unknown_prob,
                black_unknown_prob=black_unknown_prob,
                both_unknown_prob=both_unknown_prob,
                seed=seed,
                p_no=p_no,
                include_piece_qa=include_piece_qa,
                piece_sampling_probs=piece_sampling_probs,
                fen=fen,
                include_what_is_on_qa=include_what_is_on_qa,
                what_is_on_prob=what_is_on_prob,
            )
            token_ids_list.append(token_ids)

        return batch.select("split_bucket").with_columns(
            pl.Series("token_ids", token_ids_list, dtype=pl.List(pl.UInt16))
        )

    lf = lf.with_columns(
        [
            (pl.col("uci_moves").hash(seed=seed) % 1000).alias("split_bucket"),
            (pl.col("uci_moves").hash(seed=seed + 1) % 1000).alias("elo_bucket"),
        ],
    )

    row_count = lf.select(pl.len()).collect().item()
    lf.map_batches(
        build_tokens_batch,
        schema={"split_bucket": pl.UInt64, "token_ids": pl.List(pl.UInt16)},
    ).sink_parquet(output_path)
    return row_count


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


def _process_one_shard(
    parquet_path: Path,
    seed: int,
    output_path: Path,
    move_vocab_path: Path,
    piece_aware_moves: bool,
    side_prefixed_moves: bool,
    include_check_qa: bool,
    check_qa_prob: float,
    include_piece_qa: bool,
    king_base_prob: float,
    unknown_elo: dict[str, float],
    include_what_is_on_qa: bool = False,
    what_is_on_prob: float = 0.0,
) -> tuple[str, int, str]:
    load_move_vocab(
        move_vocab_path,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )
    count = process_file_streaming(
        parquet_path,
        seed,
        output_path,
        include_check_qa=include_check_qa,
        check_qa_prob=check_qa_prob,
        include_piece_qa=include_piece_qa,
        king_base_prob=king_base_prob,
        unknown_elo=unknown_elo,
        include_what_is_on_qa=include_what_is_on_qa,
        what_is_on_prob=what_is_on_prob,
    )
    return parquet_path.name, count, output_path.name


def compute_stats(tokenized_lf: pl.LazyFrame, block_size: int) -> dict[str, float]:
    seq_len_lf = tokenized_lf.select(pl.col("token_ids").list.len().alias("len"))

    stats = seq_len_lf.select(
        pl.col("len").count().alias("total"),
        pl.col("len").min().alias("min"),
        pl.col("len").max().alias("max"),
        pl.col("len").mean().alias("mean"),
        pl.col("len").median().alias("median"),
        pl.col("len").std().alias("std"),
        pl.col("len").quantile(0.95).alias("p95"),
        pl.col("len").quantile(0.99).alias("p99"),
        pl.col("len").quantile(0.999).alias("p999"),
        (pl.col("len") > block_size).sum().alias("over_block_size"),
    ).collect()
    return {
        "total": stats.item(0, "total"),
        "min": stats.item(0, "min"),
        "max": stats.item(0, "max"),
        "mean": stats.item(0, "mean"),
        "median": stats.item(0, "median"),
        "std": stats.item(0, "std"),
        "p95": stats.item(0, "p95"),
        "p99": stats.item(0, "p99"),
        "p999": stats.item(0, "p999"),
        "over_block_size": stats.item(0, "over_block_size"),
    }


def compute_token_mix_stats(tokenized_lf: pl.LazyFrame) -> dict[str, float]:
    result_ids = [WHITE_WON_ID, BLACK_WON_ID, DRAW_ID, UNKNOWN_RESULT_ID]
    elo_ids = [
        ELO_BELOW_1000_ID,
        ELO_1000_1499_ID,
        ELO_1500_1999_ID,
        ELO_2000_2499_ID,
        ELO_2500_2999_ID,
        ELO_ABOVE_3000_ID,
        ELO_UNKNOWN_ID,
    ]
    special_ids = list(SPECIAL_TOKENS.values())

    stats = (
        tokenized_lf.select(
            pl.col("token_ids").list.len().sum().alias("total_tokens"),
            pl.col("token_ids")
            .list.eval((pl.element() == IS_CHECK_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("is_check_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == YES_CHECK_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("yes_check_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == NO_CHECK_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("no_check_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == PIECE_TYPE_MOVED_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("piece_type_moved_count"),
            pl.col("token_ids")
            .list.eval(
                pl.element().is_in(list(WHATS_ON_SQUARE_TOKEN_IDS)).cast(pl.UInt32),
                parallel=True,
            )
            .list.sum()
            .sum()
            .alias("what_is_on_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == EMPTY_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("empty_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == PAWN_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("pawn_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == KNIGHT_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("knight_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == BISHOP_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("bishop_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == ROOK_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("rook_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == QUEEN_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("queen_count"),
            pl.col("token_ids")
            .list.eval((pl.element() == KING_ID).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("king_count"),
            pl.col("token_ids")
            .list.eval(pl.element().is_in(result_ids).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("result_count"),
            pl.col("token_ids")
            .list.eval(pl.element().is_in(elo_ids).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("elo_count"),
            pl.col("token_ids")
            .list.eval(pl.element().is_in(special_ids).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias("special_count"),
        )
        .collect()
        .row(0)
    )

    total_tokens = int(stats[0] or 0)
    is_check_count = int(stats[1] or 0)
    yes_check_count = int(stats[2] or 0)
    no_check_count = int(stats[3] or 0)
    piece_type_moved_count = int(stats[4] or 0)
    what_is_on_count = int(stats[5] or 0)
    empty_count = int(stats[6] or 0)
    pawn_count = int(stats[7] or 0)
    knight_count = int(stats[8] or 0)
    bishop_count = int(stats[9] or 0)
    rook_count = int(stats[10] or 0)
    queen_count = int(stats[11] or 0)
    king_count = int(stats[12] or 0)
    result_count = int(stats[13] or 0)
    elo_count = int(stats[14] or 0)
    special_count = int(stats[15] or 0)

    check_qa_count = is_check_count + yes_check_count + no_check_count
    piece_answer_count = (
        pawn_count + knight_count + bishop_count + rook_count + queen_count + king_count
    )
    piece_qa_count = piece_type_moved_count + piece_answer_count
    outcome_prefix_count = result_count + elo_count
    uci_move_count = max(0, total_tokens - special_count)

    def pct(count: int) -> float:
        return (count / total_tokens * 100.0) if total_tokens > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "uci_move_count": uci_move_count,
        "check_qa_count": check_qa_count,
        "piece_qa_count": piece_qa_count,
        "piece_answer_count": piece_answer_count,
        "outcome_prefix_count": outcome_prefix_count,
        "is_check_count": is_check_count,
        "yes_check_count": yes_check_count,
        "no_check_count": no_check_count,
        "piece_type_moved_count": piece_type_moved_count,
        "pawn_count": pawn_count,
        "knight_count": knight_count,
        "bishop_count": bishop_count,
        "rook_count": rook_count,
        "queen_count": queen_count,
        "king_count": king_count,
        "result_count": result_count,
        "elo_count": elo_count,
        "uci_move_pct": pct(uci_move_count),
        "check_qa_pct": pct(check_qa_count),
        "piece_qa_pct": pct(piece_qa_count),
        "outcome_prefix_pct": pct(outcome_prefix_count),
        "what_is_on_count": what_is_on_count,
        "empty_count": empty_count,
        "occupied_count": what_is_on_count - empty_count,
    }


def one_row_one_game(lazy_df: pl.LazyFrame, block_size: int) -> pl.LazyFrame:
    window_size = block_size + 1
    return lazy_df.select(pl.col("token_ids").list.slice(0, window_size).alias("token_ids"))


@hydra.main(version_base=None, config_path="../../config", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    piece_aware_moves = bool(cfg.get("piece_aware_moves", False))
    side_prefixed_moves = bool(cfg.get("side_prefixed_moves", True))
    block_size = int(cfg.block_size)
    seed = int(cfg.seed)

    qa = cfg.get("qa", {})

    check_cfg = qa.get("check", {})
    include_check_qa = bool(check_cfg.get("enabled", True))
    check_qa_prob = float(check_cfg.get("prob", 0.5))
    if not 0.0 <= check_qa_prob <= 1.0:
        raise ValueError(f"qa.check.prob must be in [0, 1], got {check_qa_prob}")

    piece_type_moved_cfg = qa.get("piece_type_moved", {})
    include_piece_qa = bool(piece_type_moved_cfg.get("enabled", True))
    king_base_prob = float(piece_type_moved_cfg.get("king_base_prob", 0.5))
    if not 0.0 <= king_base_prob <= 1.0:
        raise ValueError(
            f"qa.piece_type_moved.king_base_prob must be in [0, 1], got {king_base_prob}"
        )

    what_is_on_cfg = qa.get("what_is_on", {})
    include_what_is_on_qa = bool(what_is_on_cfg.get("enabled", False))
    what_is_on_prob = float(what_is_on_cfg.get("prob", 0.0))
    if not 0.0 <= what_is_on_prob <= 1.0:
        raise ValueError(f"qa.what_is_on.prob must be in [0, 1], got {what_is_on_prob}")
    unknown_elo = {
        "normal_prob": float(cfg.unknown_elo.normal_prob),
        "white_unknown_prob": float(cfg.unknown_elo.white_unknown_prob),
        "black_unknown_prob": float(cfg.unknown_elo.black_unknown_prob),
        "both_unknown_prob": float(cfg.unknown_elo.both_unknown_prob),
    }
    if abs(sum(unknown_elo.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"unknown_elo probabilities must sum to 1.0, got {sum(unknown_elo.values())}"
        )

    parquet_files = sorted(RAW_UCI_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Aix-filtered games found in {RAW_UCI_DIR}")

    PRETRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_move_vocab_from_corpus(
        parquet_files,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
        output_path=MOVE_VOCAB_PATH,
    )
    load_move_vocab(
        MOVE_VOCAB_PATH,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )

    temp_dir = PRETRAIN_DATASET_PATH.parent / "temp_preprocess"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total_games = 0
    workers_cfg = int(cfg.get("preprocess_workers", 0) or 0)
    max_workers = min(
        workers_cfg if workers_cfg > 0 else min(len(parquet_files), os.cpu_count() or 1), 8
    )
    logger.info("Processing {} shards with {} workers", len(parquet_files), max_workers)

    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {}
        for idx, parquet_path in enumerate(parquet_files):
            output_path = temp_dir / f"part_{idx:04d}.parquet"
            future = executor.submit(
                _process_one_shard,
                parquet_path,
                seed,
                output_path,
                MOVE_VOCAB_PATH,
                piece_aware_moves,
                side_prefixed_moves,
                include_check_qa,
                check_qa_prob,
                include_piece_qa,
                king_base_prob,
                unknown_elo,
                include_what_is_on_qa,
                what_is_on_prob,
            )
            futures[future] = parquet_path.name

        for future in as_completed(futures):
            parquet_name = futures[future]
            try:
                done_name, count, output_name = future.result()
                total_games += count
                logger.info("Processed {}: {} games -> {}", done_name, count, output_name)
            except Exception as e:
                logger.error("Failed to process {}: {}", parquet_name, e)
                raise

    all_parts = list(temp_dir.glob("part_*.parquet"))
    if not all_parts:
        raise RuntimeError("No data generated")

    combined_lf = pl.concat(pl.scan_parquet(p) for p in all_parts)

    stats = compute_stats(combined_lf, block_size)
    if stats["total"] == 0:
        raise RuntimeError("No games found in raw dataset.")

    logger.info(
        "Sequence length stats: total={}, min={}, max={}, mean={:.1f}, median={}, "
        "std={:.1f}, p95={}, p99={}, p999={}",
        stats["total"],
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["median"],
        stats["std"],
        stats["p95"],
        stats["p99"],
        stats["p999"],
    )

    max_tokens = block_size
    over_block_size_count = stats.get("over_block_size", 0)
    total_count = stats["total"]
    pct = over_block_size_count / total_count * 100
    logger.info(
        "Games with >{} tokens: {} ({:.2f}%) - filtering out >{}",
        max_tokens,
        over_block_size_count,
        pct,
        max_tokens,
    )

    filtered_lf = combined_lf.filter(pl.col("token_ids").list.len() <= max_tokens)

    token_mix = compute_token_mix_stats(filtered_lf.select("token_ids"))
    logger.info(
        "Token mix after >block_size filtering: UCI moves={} ({:.2f}%), "
        "check Q&A={} ({:.2f}%), piece Q&A={} ({:.2f}%), "
        "what_is_on={} ({:.2f}%), outcome prefix={} ({:.2f}%)",
        token_mix["uci_move_count"],
        token_mix["uci_move_pct"],
        token_mix["check_qa_count"],
        token_mix["check_qa_pct"],
        token_mix["piece_qa_count"],
        token_mix["piece_qa_pct"],
        token_mix["what_is_on_count"],
        (token_mix["what_is_on_count"] / token_mix["total_tokens"] * 100)
        if token_mix["total_tokens"] > 0
        else 0,
        token_mix["outcome_prefix_count"],
        token_mix["outcome_prefix_pct"],
    )
    logger.info(
        "Token mix details: <is_check>={}, <yes_check>={}, <no_check>={}, "
        "<piece_type_moved>={}, <pawn>={}, <knight>={}, <bishop>={}, <rook>={}, "
        "<queen>={}, <king>={}, result_tokens={}, elo_tokens={}, total_tokens={}",
        token_mix["is_check_count"],
        token_mix["yes_check_count"],
        token_mix["no_check_count"],
        token_mix["piece_type_moved_count"],
        token_mix["pawn_count"],
        token_mix["knight_count"],
        token_mix["bishop_count"],
        token_mix["rook_count"],
        token_mix["queen_count"],
        token_mix["king_count"],
        token_mix["result_count"],
        token_mix["elo_count"],
        token_mix["total_tokens"],
    )
    if token_mix["piece_answer_count"] > 0:
        total_piece_answers = token_mix["piece_answer_count"]
        logger.info(
            "Piece Q&A answer distribution: pawn={:.2f}%, knight={:.2f}%, bishop={:.2f}%, "
            "rook={:.2f}%, queen={:.2f}%, king={:.2f}%",
            token_mix["pawn_count"] / total_piece_answers * 100.0,
            token_mix["knight_count"] / total_piece_answers * 100.0,
            token_mix["bishop_count"] / total_piece_answers * 100.0,
            token_mix["rook_count"] / total_piece_answers * 100.0,
            token_mix["queen_count"] / total_piece_answers * 100.0,
            token_mix["king_count"] / total_piece_answers * 100.0,
        )

    train_lf = one_row_one_game(
        filtered_lf.filter(pl.col("split_bucket") != 0).select("token_ids"),
        block_size=block_size,
    )
    eval_lf = one_row_one_game(
        filtered_lf.filter(pl.col("split_bucket") == 0).select("token_ids"),
        block_size=block_size,
    )

    train_lf.sink_parquet(PRETRAIN_DATASET_PATH)
    eval_lf.sink_parquet(EVAL_DATASET_PATH)

    shutil.rmtree(temp_dir)

    train_rows = pl.scan_parquet(PRETRAIN_DATASET_PATH).select(pl.len()).collect().item()
    eval_rows = pl.scan_parquet(EVAL_DATASET_PATH).select(pl.len()).collect().item()
    if train_rows == 0:
        raise RuntimeError("Train dataset is empty. Increase input data or reduce block_size.")

    logger.info(
        "Successfully processed {} games -> {} (one-row-one-game, train rows: {}, eval rows: {})",
        stats["total"],
        PRETRAIN_DATASET_PATH.parent,
        train_rows,
        eval_rows,
    )


if __name__ == "__main__":
    main()
