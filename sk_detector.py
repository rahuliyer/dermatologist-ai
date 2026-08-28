"""Backward-compatible entry point. Prefer: python cancer_detector.py --task sk"""

import sys

from cancer_detector import main


if __name__ == "__main__":
    sys.exit(main(["--task", "sk"] + sys.argv[1:]))
