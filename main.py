import logging
import io
from pathlib import Path
from typing import Generator

import requests
import zstandard as zstd
import polars as pl
import chess.pgn

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, download_links_path: Path, output_path: Path):
        self.download_links_path = download_links_path
        self.output_path = output_path

        if not self.download_links_path.exists():
            logger.error(f"Download list file {self.download_links_path} does not exist.")
            raise FileNotFoundError(f"Download list file {self.download_links_path} does not exist.")

    def run(self):
        if not self.download_links_path.exists():
            logger.error(f"Download list file {self.download_links_path} does not exist.")
            raise FileNotFoundError(f"Download list file {self.download_links_path} does not exist.")

        urls = self.download_links_path.read_text().splitlines()
        for url in urls:
            if url.strip():
                self.process_link(url.strip())

    def process_link(self, link: str):
        # example: https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
        date = link.split("_")[-1].split(".")[0]
        base_output_path = self.output_path.with_name(f"{self.output_path.stem}_{date}")

        logger.info(f"Processing {date}")

        game_count = 0
        part_idx = 0

        with requests.get(link, stream=True) as response:
            response.raise_for_status()  # ensure we catch any HTTP errors
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(response.raw) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                for df_batch in self._parse_to_dataframe(text_stream):
                    if df_batch is None or df_batch.height == 0:
                        logger.warning(f"No valid games found in batch for {date}, skipping save for this batch.")
                        continue

                    output_path = base_output_path.with_name(f"{base_output_path.name}_part_{part_idx}.parquet")
                    df_batch.write_parquet(output_path)
                    game_count += df_batch.height
                    logger.info(f"Saved {df_batch.height} games to {output_path}. Total games for {date}: {game_count}")
                    part_idx += 1

        if game_count == 0:
            logger.warning(f"No games found in {date} matching criteria.")
        else:
            logger.info(f"Finished processing {date}. Total games saved: {game_count}")

    def _parse_to_dataframe(self, text_stream: io.TextIOWrapper) -> Generator[pl.DataFrame, None, None]:
        batch_data = []
        count = 0
        count_accepted = 0

        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            count += 1
            if count % 1000 == 0:
                logger.info(f"Parsed {count} games. Accepted {count_accepted} so far.")

            headers = game.headers

            # Filtering
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue  # Skip games with invalid Elo

            if white_elo < config.MIN_ELO or black_elo < config.MIN_ELO:
                continue

            if abs(white_elo - black_elo) > config.MAX_ELO_DIFF:
                logger.info(f"Skipping game with Elo difference {abs(white_elo - black_elo)} which exceeds criteria.")
                continue

            result = headers.get("Result", "*")
            if result not in ["1-0", "0-1"] + (["1/2-1/2"] if config.EXCLUDE_DRAWS else []):
                logger.info(f"Skipping game with result {result} which does not meet criteria.")
                continue  # Skip aborted/unknown games

            base, _, increment = headers.get("TimeControl", "").partition("+")
            if int(base) < config.MIN_BASE_TIME:
                continue

            moves = [move.uci() for move in game.mainline_moves()]
            ply_count = len(moves)

            if ply_count < config.MIN_MOVES:
                continue

            batch_data.append(
                {
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "result": result,
                    "moves": str(moves),
                    "opening": headers.get("Opening", ""),
                }
            )

            count_accepted += 1

            if len(batch_data) >= config.BATCH_SIZE:
                yield pl.DataFrame(batch_data)
                batch_data = []

        if batch_data:
            yield pl.DataFrame(batch_data)


if __name__ == "__main__":
    download_links_path = Path("data/download_links.txt")
    output_path = Path("data/processed_games.parquet")

    pipeline = DataPipeline(download_links_path, output_path)
    pipeline.run()
