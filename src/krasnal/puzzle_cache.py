from pathlib import Path


def source_game_cache_path_for(
    puzzle_path: Path,
    *,
    sample_size: int | None,
    seed: int,
) -> Path:
    sample_label = "all" if sample_size is None or sample_size <= 0 else str(sample_size)
    puzzle_path = Path(puzzle_path)
    return puzzle_path.with_name(
        f"{puzzle_path.stem}_source_cache_seed{seed}_n{sample_label}.jsonl"
    )
