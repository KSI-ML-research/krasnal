import sys

from loguru import logger


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


configure_logging()
