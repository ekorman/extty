"""Structured logging for extty with TUI-matched cyberpunk colors."""

from __future__ import annotations

import logging
import os
import sys
from typing import IO

# Cyberpunk palette mirroring tui/src/main.rs (24-bit truecolor escapes).
_NEON_CYAN = "\033[38;2;0;255;255m"
_NEON_MAGENTA = "\033[38;2;255;0;128m"
_NEON_GREEN = "\033[38;2;0;255;136m"
_NEON_YELLOW = "\033[38;2;255;255;0m"
_DIM_CYAN = "\033[38;2;0;139;139m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: _DIM_CYAN,
    logging.INFO: _NEON_CYAN,
    logging.WARNING: _NEON_YELLOW,
    logging.ERROR: _NEON_MAGENTA,
    logging.CRITICAL: _BOLD + _NEON_MAGENTA,
}

_PLAIN_FMT = "%(asctime)s | extty | %(levelname)s | %(message)s"
_COLOR_FMT = (
    f"{_DIM_CYAN}%(asctime)s{_RESET} "
    f"{_DIM_CYAN}|{_RESET} "
    f"{_NEON_GREEN}extty{_RESET} "
    f"{_DIM_CYAN}|{_RESET} "
    f"%(levelname)s "
    f"{_DIM_CYAN}|{_RESET} "
    f"%(message)s"
)
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _supports_color(stream: IO[str]) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("EXTTY_FORCE_COLOR"):
        return True
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


class _ColorFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool) -> None:
        super().__init__(
            fmt=_COLOR_FMT if use_color else _PLAIN_FMT,
            datefmt=_DATEFMT,
        )
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_color:
            return super().format(record)
        original = record.levelname
        color = _LEVEL_COLORS.get(record.levelno, "")
        record.levelname = f"{color}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def _configure() -> logging.Logger:
    logger = logging.getLogger("extty")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColorFormatter(use_color=_supports_color(handler.stream)))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


log = _configure()
