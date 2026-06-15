"""Logging helpers shared by command line scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: Path, name: str = "diabetes_prediction") -> logging.Logger:
    """Configure console and file logging for a script run."""

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger
    except Exception as exc:
        logging.basicConfig(level=logging.INFO)
        fallback_logger = logging.getLogger(name)
        fallback_logger.exception("Failed to initialize file logging: %s", exc)
        return fallback_logger
