import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import bulletchess
import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

import wandb
from krasnal.config import (
    CLOCK_IGNORE_ID,
    EVAL_DATASET_PATH,
    MOVE_VOCAB_PATH,
    PRETRAIN_DATASET_PATH,
    RAW_UCI_DIR,
)
from krasnal.sampling import sample_bool, whats_on_square_index
from krasnal.tokens import (
    BLACK_WON_ID,
    COLORED_PIECE_TOKENS,
    DRAW_ID,
    ELO_TOKENS,
    EMPTY_ID,
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    NO_CHECK_ID,
    SPECIAL_TOKENS,
    TC_TOKENS,
    UNKNOWN_RESULT_ID,
    WHATS_ON_SQUARE,
    WHATS_ON_SQUARE_TOKEN_IDS,
    WHITE_WON_ID,
    YES_CHECK_ID,
    get_elo_bucket,
    get_move_clock_pairs,
    get_moves_only,
    get_time_control_bucket,
    load_move_vocab,
    move_key_for_ply,
    move_token_id_for_ply,
    normalize_piece_type,
    result_to_token_id,
    save_move_vocab,
)


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
) -> tuple[list[tuple[int, int]], bool]:
    if clocks_white is None or clocks_black is None:
        return [(CLOCK_IGNORE_ID, CLOCK_IGNORE_ID)] * len(moves_list), False
    if not isinstance(clocks_white, list) or not isinstance(clocks_black, list):
        raise ValueError(f"{context}: clocks_white and clocks_black must be lists")

    expected_white = (len(moves_list) + 1) // 2
    expected_black = len(moves_list) // 2
    if len(clocks_white) != expected_white or len(clocks_black) != expected_black:
        raise ValueError(
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
    return clock_pairs, True


def _build_game_tokens(
    uci_moves: str,
    is_check: list[bool],
    piece_moved: list[str],
    result: str,
    white_rating: int,
    black_rating: int,
    time_initial: int | None,
    time_increment: int | None,
    time_control_enabled: bool,
    include_check_qa: bool,
    check_qa_prob: float,
    seed: int,
    p_no: float,
    clocks_white: list[int] | None = None,
    clocks_black: list[int] | None = None,
    fen: str | None = None,
    include_what_is_on_qa: bool = False,
    what_is_on_prob: float = 0.0,
) -> tuple[list[int], list[int], list[int]]:
    if not uci_moves:
        return [], [], []

    if include_what_is_on_qa:
        b = bulletchess.Board.from_fen(fen) if fen else bulletchess.Board()

    moves_list = uci_moves.split()
    piece_types = _validated_piece_moved(
        piece_moved,
        moves_list,
        context=f"game {uci_moves[:80]!r}",
    )
    move_clock_pairs, has_clocks = _validated_clock_arrays(
        clocks_white,
        clocks_black,
        moves_list,
        time_initial=time_initial,
        context=f"game {uci_moves[:80]!r}",
    )

    result_tokens = []
    active_clock_ids = []
    opponent_clock_ids = []

    def append_token(
        token_id: int,
        active_clock_id: int = CLOCK_IGNORE_ID,
        opponent_clock_id: int = CLOCK_IGNORE_ID,
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

        if include_check_qa:
            gives_check = ply < len(is_check) and bool(is_check[ply])
            if gives_check:
                if sample_bool(seed=seed, game_key=uci_moves, ply=ply, probability=check_qa_prob):
                    append_token(IS_CHECK_ID)
                    append_token(YES_CHECK_ID)
            elif sample_bool(seed=seed, game_key=uci_moves, ply=ply, probability=p_no):
                append_token(IS_CHECK_ID)
                append_token(NO_CHECK_ID)

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

                append_token(whats_on_token_id)
                append_token(ans_id)

    white_elo = get_elo_bucket(white_rating)
    black_elo = get_elo_bucket(black_rating)

    prefix_tokens = [
        GAME_START_ID,
    ]
    if time_control_enabled:
        prefix_tokens.append(get_time_control_bucket(time_initial, time_increment))
    prefix_tokens.extend([result_to_token_id(result), white_elo, black_elo])
    prefix_clocks = [CLOCK_IGNORE_ID] * len(prefix_tokens)
    token_ids = prefix_tokens + result_tokens + [GAME_END_ID]
    active_ids = prefix_clocks + active_clock_ids + [CLOCK_IGNORE_ID]
    opponent_ids = prefix_clocks + opponent_clock_ids + [CLOCK_IGNORE_ID]
    if has_clocks and not (len(token_ids) == len(active_ids) == len(opponent_ids)):
        raise RuntimeError("Clock/token alignment failed")
    return token_ids, active_ids, opponent_ids


def process_file_streaming(
    parquet_path: Path,
    seed: int,
    output_path: Path,
    include_check_qa: bool,
    check_qa_prob: float,
    time_control_enabled: bool,
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

    def build_tokens_batch(batch: pl.DataFrame) -> pl.DataFrame:
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
        if time_control_enabled:
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
            token_ids, active_clock_ids, opponent_clock_ids = _build_game_tokens(
                uci_moves=uci_moves,
                is_check=is_check,
                piece_moved=piece_moved,
                result=result,
                white_rating=white_rating,
                black_rating=black_rating,
                time_initial=time_initial,
                time_increment=time_increment,
                time_control_enabled=time_control_enabled,
                include_check_qa=include_check_qa,
                check_qa_prob=check_qa_prob,
                seed=seed,
                p_no=p_no,
                clocks_white=clocks_white,
                clocks_black=clocks_black,
                fen=fen,
                include_what_is_on_qa=include_what_is_on_qa,
                what_is_on_prob=what_is_on_prob,
            )
            token_ids_list.append(token_ids)
            active_clock_ids_list.append(active_clock_ids)
            opponent_clock_ids_list.append(opponent_clock_ids)

        return batch.select("split_bucket").with_columns(
            pl.Series("token_ids", token_ids_list, dtype=pl.List(pl.UInt16)),
            pl.Series("active_clock_ids", active_clock_ids_list, dtype=pl.List(pl.UInt32)),
            pl.Series("opponent_clock_ids", opponent_clock_ids_list, dtype=pl.List(pl.UInt32)),
        )

    lf = lf.with_columns((pl.col("uci_moves").hash(seed=seed) % 1000).alias("split_bucket"))

    row_count = lf.select(pl.len()).collect().item()
    lf.map_batches(
        build_tokens_batch,
        schema={
            "split_bucket": pl.UInt64,
            "token_ids": pl.List(pl.UInt16),
            "active_clock_ids": pl.List(pl.UInt32),
            "opponent_clock_ids": pl.List(pl.UInt32),
        },
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
    time_control_enabled: bool,
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
        time_control_enabled=time_control_enabled,
        include_what_is_on_qa=include_what_is_on_qa,
        what_is_on_prob=what_is_on_prob,
    )
    return parquet_path.name, count, output_path.name


def _seq_len_stats_from_lf(seq_len_lf: pl.LazyFrame, block_size: int) -> dict[str, float]:
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


def _token_mix_raw_sums(tokenized_lf: pl.LazyFrame) -> tuple[int, ...]:
    result_ids = [WHITE_WON_ID, BLACK_WON_ID, DRAW_ID, UNKNOWN_RESULT_ID]
    elo_ids = list(ELO_TOKENS.values())
    tc_ids = list(TC_TOKENS.values())
    special_ids = list(SPECIAL_TOKENS.values())

    exprs = [
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
        .list.eval(
            pl.element().is_in(list(WHATS_ON_SQUARE_TOKEN_IDS)).cast(pl.UInt32), parallel=True
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
        .list.eval(
            pl.element().is_in(list(COLORED_PIECE_TOKENS.values())).cast(pl.UInt32), parallel=True
        )
        .list.sum()
        .sum()
        .alias("piece_answer_count"),
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
        .list.eval(pl.element().is_in(tc_ids).cast(pl.UInt32), parallel=True)
        .list.sum()
        .sum()
        .alias("tc_count"),
        pl.col("token_ids")
        .list.eval(pl.element().is_in(special_ids).cast(pl.UInt32), parallel=True)
        .list.sum()
        .sum()
        .alias("special_count"),
        pl.col("token_ids")
        .list.eval((pl.element() == GAME_START_ID).cast(pl.UInt32), parallel=True)
        .list.sum()
        .sum()
        .alias("game_start_count"),
        pl.col("token_ids")
        .list.eval((pl.element() == GAME_END_ID).cast(pl.UInt32), parallel=True)
        .list.sum()
        .sum()
        .alias("game_end_count"),
    ]
    for bucket_name, bucket_id in ELO_TOKENS.items():
        exprs.append(
            pl.col("token_ids")
            .list.eval((pl.element() == bucket_id).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias(f"elo_{bucket_name}_count")
        )
    for bucket_name, bucket_id in TC_TOKENS.items():
        exprs.append(
            pl.col("token_ids")
            .list.eval((pl.element() == bucket_id).cast(pl.UInt32), parallel=True)
            .list.sum()
            .sum()
            .alias(f"tc_{bucket_name}_count")
        )

    stats = tokenized_lf.select(*exprs).collect().row(0)
    return tuple(int(x or 0) for x in stats)


def _merge_token_mix_raw(
    acc: tuple[int, ...] | None,
    part: tuple[int, ...],
) -> tuple[int, ...]:
    if acc is None:
        return part
    return tuple(a + b for a, b in zip(acc, part, strict=True))


def _token_mix_from_raw_sums(stats: tuple[int, ...]) -> dict[str, float]:
    total_tokens = stats[0]
    is_check_count = stats[1]
    yes_check_count = stats[2]
    no_check_count = stats[3]
    what_is_on_count = stats[4]
    empty_count = stats[5]
    piece_answer_count = stats[6]
    result_count = stats[7]
    elo_count = stats[8]
    tc_count = stats[9]
    special_count = stats[10]
    game_start_count = stats[11]
    game_end_count = stats[12]

    check_qa_count = is_check_count + yes_check_count + no_check_count
    outcome_prefix_count = result_count + elo_count + tc_count
    uci_move_count = max(0, total_tokens - special_count)
    whats_on_answer_count = empty_count + piece_answer_count

    def pct(count: int) -> float:
        return (count / total_tokens * 100.0) if total_tokens > 0 else 0.0

    result = {
        "total_tokens": total_tokens,
        "uci_move_count": uci_move_count,
        "check_qa_count": check_qa_count,
        "outcome_prefix_count": outcome_prefix_count,
        "is_check_count": is_check_count,
        "yes_check_count": yes_check_count,
        "no_check_count": no_check_count,
        "result_count": result_count,
        "elo_count": elo_count,
        "tc_count": tc_count,
        "uci_move_pct": pct(uci_move_count),
        "check_qa_pct": pct(check_qa_count),
        "outcome_prefix_pct": pct(outcome_prefix_count),
        "what_is_on_count": what_is_on_count,
        "what_is_on_pct": pct(what_is_on_count),
        "empty_count": empty_count,
        "empty_pct": pct(empty_count),
        "piece_answer_count": piece_answer_count,
        "piece_answer_pct": pct(piece_answer_count),
        "whats_on_answer_count": whats_on_answer_count,
        "whats_on_answer_pct": pct(whats_on_answer_count),
        "game_start_count": game_start_count,
        "game_end_count": game_end_count,
        "game_start_pct": pct(game_start_count),
        "game_end_pct": pct(game_end_count),
    }

    idx = 13
    for bucket_name in ELO_TOKENS:
        result[f"elo_{bucket_name}_count"] = float(stats[idx])
        idx += 1
    for bucket_name in TC_TOKENS:
        result[f"tc_{bucket_name}_count"] = float(stats[idx])
        idx += 1

    return result


def compute_token_mix_stats(tokenized_lf: pl.LazyFrame) -> dict[str, float]:
    return _token_mix_from_raw_sums(_token_mix_raw_sums(tokenized_lf))


def one_row_one_game(lazy_df: pl.LazyFrame, block_size: int) -> pl.LazyFrame:
    window_size = block_size + 1
    columns = ["token_ids", "active_clock_ids", "opponent_clock_ids"]
    return lazy_df.select(
        [pl.col(column).list.slice(0, window_size).alias(column) for column in columns]
    )


def _chunk_paths(paths: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size < 1:
        raise ValueError(f"preprocess_concat_batch_size must be >= 1, got {chunk_size}")
    return [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]


def _log_preprocess_to_wandb(
    *,
    cfg: DictConfig,
    token_mix: dict[str, float],
    seq_stats: dict[str, float],
    total_games: int,
    train_rows: int,
    eval_rows: int,
    over_block_size_count: int,
) -> None:
    """Log dataset statistics to a W&B run tagged 'preprocess'."""
    project = str(cfg.get("wandb_project", "uwr-ksai/krasnal"))
    wandb.init(project=project, job_type="preprocess", tags=["preprocess"])

    wandb.summary["dataset/total_games"] = total_games
    wandb.summary["dataset/train_rows"] = train_rows
    wandb.summary["dataset/eval_rows"] = eval_rows
    wandb.summary["dataset/removed_over_block_size"] = over_block_size_count
    wandb.summary["dataset/removed_over_block_size_pct"] = (
        over_block_size_count / total_games * 100.0 if total_games > 0 else 0.0
    )

    wandb.summary["dataset/seq_len_min"] = seq_stats["min"]
    wandb.summary["dataset/seq_len_max"] = seq_stats["max"]
    wandb.summary["dataset/seq_len_mean"] = seq_stats["mean"]
    wandb.summary["dataset/seq_len_p95"] = seq_stats["p95"]
    wandb.summary["dataset/seq_len_p99"] = seq_stats["p99"]

    wandb.summary["dataset/total_tokens"] = token_mix["total_tokens"]
    wandb.summary["dataset/uci_move_pct"] = token_mix["uci_move_pct"]
    wandb.summary["dataset/check_qa_pct"] = token_mix["check_qa_pct"]
    wandb.summary["dataset/what_is_on_pct"] = token_mix["what_is_on_pct"]
    wandb.summary["dataset/whats_on_answer_pct"] = token_mix["whats_on_answer_pct"]
    wandb.summary["dataset/outcome_prefix_pct"] = token_mix["outcome_prefix_pct"]
    wandb.summary["dataset/game_start_pct"] = token_mix["game_start_pct"]
    wandb.summary["dataset/game_end_pct"] = token_mix["game_end_pct"]

    total_elo = sum(token_mix.get(f"elo_{b}_count", 0) for b in ELO_TOKENS)
    for bucket_name in ELO_TOKENS:
        count = token_mix.get(f"elo_{bucket_name}_count", 0)
        pct = (count / total_elo * 100.0) if total_elo > 0 else 0.0
        wandb.summary[f"dataset/elo/{bucket_name}"] = pct

    total_tc = sum(token_mix.get(f"tc_{b}_count", 0) for b in TC_TOKENS)
    for bucket_name in TC_TOKENS:
        count = token_mix.get(f"tc_{bucket_name}_count", 0)
        pct = (count / total_tc * 100.0) if total_tc > 0 else 0.0
        wandb.summary[f"dataset/tc/{bucket_name}"] = pct

    wandb.finish()
    logger.info("Preprocessing statistics logged to W&B project '{}'", project)


_THRESHOLD = 30


def _bucket(seconds: int) -> str:
    if seconds < 10:
        return "<10s"
    if seconds < 30:
        return "10-30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    if seconds < 300:
        return "120-300s"
    return ">300s"


_BUCKET_LABELS = ("<10s", "10-30s", "30-60s", "60-120s", "120-300s", ">300s")


def run_clock_report(path: Path = EVAL_DATASET_PATH, threshold: int = _THRESHOLD) -> None:
    df = (
        pl.scan_parquet(path)
        .select(pl.col("token_ids", "active_clock_ids", "opponent_clock_ids"))
        .collect()
    )

    total_games = len(df)
    total_plies = 0
    plies_both = 0
    games_with_clock = 0
    plies_low_active = 0
    plies_low_both = 0
    buckets = {label: 0 for label in _BUCKET_LABELS}

    for row in df.iter_rows(named=True):
        token_ids = [int(x) for x in row["token_ids"]]
        act = [int(x) for x in row["active_clock_ids"]]
        opp = [int(x) for x in row["opponent_clock_ids"]]
        moves = get_moves_only(token_ids)
        pairs = get_move_clock_pairs(token_ids, act, opp)

        if pairs is None or len(pairs) != len(moves):
            continue

        game_has_clock = False
        for a, o in pairs:
            total_plies += 1
            a_ok = a != CLOCK_IGNORE_ID
            o_ok = o != CLOCK_IGNORE_ID

            if a_ok and o_ok:
                plies_both += 1
                game_has_clock = True

            if a_ok:
                if a < threshold:
                    plies_low_active += 1
                    if o_ok and o < threshold:
                        plies_low_both += 1
                buckets[_bucket(a)] += 1

        if game_has_clock:
            games_with_clock += 1

    logger.info(f"Clock Report — {path}")
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total move plies: {total_plies}")

    pct_both = 100.0 * plies_both / total_plies if total_plies else 0.0
    pct_games = 100.0 * games_with_clock / total_games if total_games else 0.0
    logger.info("Clock Coverage:")
    logger.info(f"  positions with both clocks known:  {pct_both:.2f}% ({plies_both})")
    logger.info(f"  games with any clock data:         {pct_games:.2f}% ({games_with_clock})")

    pct_low_clocked = 100.0 * plies_low_active / plies_both if plies_both else 0.0
    pct_low_all = 100.0 * plies_low_active / total_plies if total_plies else 0.0
    pct_low_both = 100.0 * plies_low_both / plies_both if plies_both else 0.0
    logger.info(f"Low Time (<{threshold}s side to move):")
    logger.info(
        f"  among clocked positions:            {pct_low_clocked:.2f}% ({plies_low_active})"
    )
    logger.info(f"  among all positions:                {pct_low_all:.2f}% ({plies_low_active})")
    logger.info(
        f"  both players <{threshold}s:                  {pct_low_both:.2f}% ({plies_low_both})"
    )

    plies_with_active = sum(buckets.values())
    logger.info("Clock Distribution (active clock):")
    for label in _BUCKET_LABELS:
        pct = 100.0 * buckets[label] / plies_with_active if plies_with_active else 0.0
        logger.info(f"  {label:>8s}: {pct:.2f}%")


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

    what_is_on_cfg = qa.get("what_is_on", {})
    include_what_is_on_qa = bool(what_is_on_cfg.get("enabled", False))
    what_is_on_prob = float(what_is_on_cfg.get("prob", 0.0))
    if not 0.0 <= what_is_on_prob <= 1.0:
        raise ValueError(f"qa.what_is_on.prob must be in [0, 1], got {what_is_on_prob}")

    time_control_cfg = cfg.get("time_control", {})
    time_control_enabled = bool(time_control_cfg.get("enabled", True))

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
    max_workers = int(cfg.preprocess_workers)
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
                time_control_enabled,
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

    all_parts = sorted(temp_dir.glob("part_*.parquet"))
    if not all_parts:
        raise RuntimeError("No data generated")

    concat_batch_size = max(1, int(cfg.get("preprocess_concat_batch_size", 10)))
    train_batches_dir = temp_dir / "train_batches"
    eval_batches_dir = temp_dir / "eval_batches"
    shutil.rmtree(train_batches_dir, ignore_errors=True)
    shutil.rmtree(eval_batches_dir, ignore_errors=True)
    train_batches_dir.mkdir(parents=True)
    eval_batches_dir.mkdir(parents=True)

    max_tokens = block_size
    len_chunks: list[pl.DataFrame] = []
    mix_raw: tuple[int, ...] | None = None

    for batch_idx, batch_paths in enumerate(_chunk_paths(all_parts, concat_batch_size)):
        shard_lf = pl.concat(pl.scan_parquet(p) for p in batch_paths)
        len_chunks.append(shard_lf.select(pl.col("token_ids").list.len().alias("len")).collect())
        filtered_lf = shard_lf.filter(pl.col("token_ids").list.len() <= max_tokens)
        mix_raw = _merge_token_mix_raw(
            mix_raw,
            _token_mix_raw_sums(filtered_lf.select("token_ids")),
        )
        train_lf = one_row_one_game(
            filtered_lf.filter(pl.col("split_bucket") != 0).select(
                "token_ids",
                "active_clock_ids",
                "opponent_clock_ids",
            ),
            block_size=block_size,
        )
        eval_lf = one_row_one_game(
            filtered_lf.filter(pl.col("split_bucket") == 0).select(
                "token_ids",
                "active_clock_ids",
                "opponent_clock_ids",
            ),
            block_size=block_size,
        )
        train_lf.sink_parquet(train_batches_dir / f"{batch_idx:04d}.parquet")
        eval_lf.sink_parquet(eval_batches_dir / f"{batch_idx:04d}.parquet")

    seq_lens = pl.concat(len_chunks, how="vertical")
    stats = _seq_len_stats_from_lf(seq_lens.lazy(), block_size)
    if stats["total"] == 0:
        raise RuntimeError("No games found in raw dataset.")

    logger.info(
        "Sequence length stats: min={}, max={}, mean={:.1f}, p95={}, p99={}, p999={}",
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["p95"],
        stats["p99"],
        stats["p999"],
    )

    over_block_size_count = stats.get("over_block_size", 0)
    total_count = stats["total"]
    pct_long = over_block_size_count / total_count * 100
    logger.info(
        "Filtering games with >{} tokens done: removed {} games ({:.2f}%)",
        max_tokens,
        over_block_size_count,
        pct_long,
    )

    if mix_raw is None:
        raise RuntimeError("Token mix aggregation failed")
    token_mix = _token_mix_from_raw_sums(mix_raw)
    logger.info("Token distribution:")
    logger.info("  total: 100.00% ({})", token_mix["total_tokens"])
    logger.info(
        "  moves: {:.2f}% ({})",
        token_mix["uci_move_pct"],
        token_mix["uci_move_count"],
    )
    logger.info(
        "  qa_is_check: {:.2f}% ({})",
        token_mix["check_qa_pct"],
        token_mix["check_qa_count"],
    )
    logger.info(
        "  qa_whats_on_prompt: {:.2f}% ({})",
        token_mix["what_is_on_pct"],
        token_mix["what_is_on_count"],
    )
    logger.info(
        "  qa_whats_on_answer_empty: {:.2f}% ({})",
        token_mix["empty_pct"],
        token_mix["empty_count"],
    )
    logger.info(
        "  qa_whats_on_answer_piece: {:.2f}% ({})",
        token_mix["piece_answer_pct"],
        token_mix["piece_answer_count"],
    )
    logger.info(
        "  conditioning_prefix: {:.2f}% ({})",
        token_mix["outcome_prefix_pct"],
        token_mix["outcome_prefix_count"],
    )
    logger.info(
        "  game_start: {:.2f}% ({})",
        token_mix["game_start_pct"],
        token_mix["game_start_count"],
    )
    logger.info(
        "  game_end: {:.2f}% ({})",
        token_mix["game_end_pct"],
        token_mix["game_end_count"],
    )

    logger.info("ELO Bucket Distribution:")
    total_elo = sum(token_mix[f"elo_{b}_count"] for b in ELO_TOKENS)
    if total_elo > 0:
        for bucket_name in ELO_TOKENS:
            count = token_mix[f"elo_{bucket_name}_count"]
            pct = (count / total_elo) * 100.0
            logger.info("  {}: {:.2f}%", bucket_name, pct)

    logger.info("Time Control Bucket Distribution:")
    total_tc = sum(token_mix[f"tc_{b}_count"] for b in TC_TOKENS)
    if total_tc > 0:
        for bucket_name in TC_TOKENS:
            count = token_mix[f"tc_{bucket_name}_count"]
            pct = (count / total_tc) * 100.0
            logger.info("  {}: {:.2f}%", bucket_name, pct)

    token_mix_path = PRETRAIN_DATASET_PATH.parent / "token_mix_stats.json"
    with token_mix_path.open("w") as f:
        json.dump(token_mix, f, indent=2, sort_keys=True)
        f.write("\n")

    train_parts = sorted(train_batches_dir.glob("*.parquet"))
    eval_parts = sorted(eval_batches_dir.glob("*.parquet"))
    pl.concat(pl.scan_parquet(p) for p in train_parts).sink_parquet(PRETRAIN_DATASET_PATH)
    pl.concat(pl.scan_parquet(p) for p in eval_parts).sink_parquet(EVAL_DATASET_PATH)

    shutil.rmtree(temp_dir)

    train_rows = pl.scan_parquet(PRETRAIN_DATASET_PATH).select(pl.len()).collect().item()
    eval_rows = pl.scan_parquet(EVAL_DATASET_PATH).select(pl.len()).collect().item()
    if train_rows == 0:
        raise RuntimeError("Train dataset is empty. Increase input data or reduce block_size.")

    logger.info(
        "Successfully processed {} games -> {} (train rows: {}, eval rows: {})",
        stats["total"],
        PRETRAIN_DATASET_PATH.parent,
        train_rows,
        eval_rows,
    )

    run_clock_report(EVAL_DATASET_PATH)

    _log_preprocess_to_wandb(
        cfg=cfg,
        token_mix=token_mix,
        seq_stats=stats,
        total_games=stats["total"],
        train_rows=train_rows,
        eval_rows=eval_rows,
        over_block_size_count=over_block_size_count,
    )


if __name__ == "__main__":
    main()
