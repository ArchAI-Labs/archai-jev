"""Small helpers shared by the demos. They only make the output pretty.

Every call to ``archai_jev`` itself is written out in the demo files, on purpose: this
module is the "chrome" (download progress, titles, bars), not the lesson.

Nothing here is part of the library, and the output is plain ASCII so it prints correctly
on any console (including old Windows ones).
"""

from __future__ import annotations

import logging
import re
import sys

_PROGRESS = re.compile(r": (\d+) of (\d+) bytes$")


class _DownloadProgress(logging.Handler):
    """Turns the library's log lines into a one-line download progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        match = _PROGRESS.search(message)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            sys.stdout.write(f"\r  downloading the model... {100 * done / total:3.0f}%")
            sys.stdout.flush()
        elif message.startswith("downloading"):
            print("  first run: the model files are downloaded once and verified")
        elif message.endswith(": done"):
            sys.stdout.write("\n")


def show_download_progress() -> None:
    """Print a progress line while the model is downloaded (the first run only).

    The library logs to the standard ``logging`` logger called ``archai_jev``; at level
    INFO it reports each download chunk. We listen to it and draw a tiny progress line.
    """
    logger = logging.getLogger("archai_jev")
    logger.setLevel(logging.INFO)
    logger.addHandler(_DownloadProgress())
    logger.propagate = False  # we print the model's notice ourselves, once


def title(text: str) -> None:
    """A big heading."""
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def section(text: str) -> None:
    """A smaller heading."""
    print()
    print(f"--- {text}")


def bar(probability: float, width: int = 20) -> str:
    """A text bar for a probability, for example ``[#####---------------]``."""
    filled = round(max(0.0, min(1.0, probability)) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"
