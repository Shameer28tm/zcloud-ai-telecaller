"""Tests for logging configuration."""

import logging
from zcloud_ai_telecaller.logging_config import setup_logging


def test_setup_logging():
    """Verify logging configuration returns proper logger instance and level."""
    logger = setup_logging("DEBUG")
    assert logger.name == "zcloud_ai_telecaller"
    assert logger.level == logging.DEBUG

    logger_info = setup_logging("INFO")
    assert logger_info.level == logging.INFO
