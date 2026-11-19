"""Module entrypoint for ``python -m paddleocr`` and the ``paddleocr`` console script."""

from __future__ import annotations

import os
import sys

from ._cli import main


def console_entry() -> None:
    """Execute the PaddleOCR CLI and exit cleanly on broken pipes or non-zero status."""
    try:
        code = main()
        sys.stdout.flush()
        sys.stderr.flush()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    console_entry()
