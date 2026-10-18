"""Application entry point: ``python -m microbiome``."""

import sys

from .ui.app import run_demo


def main() -> int:
    run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
