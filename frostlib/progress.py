"""Elapsed-time progress logging.

Long steps (downloads, folds, decoding) all print flushed, timestamped progress
so silence never looks like a hang. One timer serves both formats used in the
project: a running seconds counter and a mm:ss clock.
"""

from __future__ import annotations

import time


class Timer:
    """A monotonic stopwatch that prints flushed, timestamped progress lines."""

    def __init__(self, style: str = "seconds") -> None:
        if style not in ("seconds", "clock"):
            raise ValueError(f"unknown style {style!r} (use 'seconds' or 'clock')")
        self.style = style
        self.t0 = time.monotonic()

    def reset(self) -> None:
        self.t0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def stamp(self) -> str:
        elapsed = self.elapsed()
        if self.style == "clock":
            mins, secs = divmod(elapsed, 60)
            return f"[{int(mins):02d}:{secs:04.1f}]"
        return f"[{elapsed:5.1f}s]"

    def log(self, msg: str) -> None:
        print(f"{self.stamp()} {msg}", flush=True)
