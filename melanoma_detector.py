"""Backward-compatible entry point for melanoma-only training."""

import sys

from cancer_detector import main

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--task", "melanoma"]))
