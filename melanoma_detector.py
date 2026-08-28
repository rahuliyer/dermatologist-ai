"""Backward-compatible entry point. Prefer: python cancer_detector.py --task melanoma"""

import sys

from cancer_detector import main


if __name__ == "__main__":
    sys.exit(main(["--task", "melanoma"] + sys.argv[1:]))
