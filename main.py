"""Compatibility launcher for running Private Agent from a source checkout."""

import sys
from pathlib import Path

PACKAGE_SOURCE = Path(__file__).resolve().parent / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))


def main():
    from private_agent.cli import main as package_main

    package_main()


if __name__ == "__main__":
    main()
