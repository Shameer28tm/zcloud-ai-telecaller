"""Logging configuration for Z Cloud AI Telecaller."""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configures structured standard logging for the agent process."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Reset any existing handlers on root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger = logging.getLogger("zcloud_ai_telecaller")
    logger.setLevel(numeric_level)
    return logger
