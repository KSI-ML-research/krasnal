import logging
import io
from pathlib import Path

import requests
import zstandard as zstd
import polars as pl
import chess.pgn

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
        output_path = self.output_path.with_name(f"{self.output_path.stem}_{date}.parquet")

        logger.info(f"Processing {date}")

        with requests.get(link, stream=True) as response:
            response.raise_for_status()  # ensure we catch any HTTP errors
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(response.raw) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                df = self._parse_to_dataframe(text_stream)

                if df.height > 0:
                    df.write_parquet(output_path)
                    logger.info(f"Saved {df.height} games to {output_path}")
                else:
                    logger.warning(f"No games found in {date}, skipping save.")

    def _parse_to_dataframe(self, text_stream: io.TextIOWrapper) -> pl.DataFrame:
        data = []
        count = 0

        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            headers = game.headers
            data.append(
                {
                    "white_elo": headers.get("WhiteElo", 0),
                    "black_elo": headers.get("BlackElo", 0),
                    "result": headers.get("Result", "*"),  # "1-0", "0-1", "1/2-1/2", "*" (unknown)
                    "moves": str(game.mainline_moves()),
                    "opening": headers.get("Opening", ""),
                    "time_control": headers.get("TimeControl", ""),  # max time for the game
                    "termination": headers.get("Termination", ""),  # how the game ended
                }
            )

            count += 1
            if count % 1000 == 0:
                logger.info(f"Parsed {count} games...")

        return pl.DataFrame(data)


if __name__ == "__main__":
    download_list_path = Path("data/download_links.txt")
    pipeline = DataPipeline(download_list_path, Path("data/raw")).run()
