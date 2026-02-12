from dataclasses import dataclass


@dataclass
class Config:
    # Exclude games that don't meet these criteria
    MIN_ELO: int = 1800
    MIN_MOVES: int = 20
    MAX_ELO_DIFF: int = 500
    MIN_BASE_TIME: int = 300  # in seconds, e.g. 5 minutes
    EXCLUDE_DRAWS: bool = True

    # Internal pipeline config
    BATCH_SIZE: int = 50_000


config = Config()
