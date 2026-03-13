"""Simple structured logging helpers with step timing."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once for UTF-8-safe console output."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return configured logger."""
    setup_logging()
    return logging.getLogger(name)


@contextmanager
def timed_step(logger: logging.Logger, step: str, description: str) -> Iterator[None]:
    """Log start/end timestamps and elapsed seconds for a major step."""

    start_perf = time.perf_counter()
    start_ts = _utc_now()
    logger.info("STEP_START | step=%s | start=%s | description=%s", step, start_ts, description)
    try:
        yield
    finally:
        end_perf = time.perf_counter()
        end_ts = _utc_now()
        elapsed = end_perf - start_perf
        logger.info("STEP_END | step=%s | end=%s | elapsed_seconds=%.3f", step, end_ts, elapsed)
