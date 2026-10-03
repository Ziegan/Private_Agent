"""Generate or verify requirements.txt from pyproject.toml dependencies."""

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILE = ROOT / "pyproject.toml"
REQUIREMENTS_FILE = ROOT / "requirements.txt"


def generated_requirements() -> str:
    with PROJECT_FILE.open("rb") as project_file:
        metadata = tomllib.load(project_file)
    dependencies = metadata["project"]["dependencies"]
    return (
        "# Generated from [project].dependencies in pyproject.toml "
        "by scripts/sync_requirements.py.\n"
        + "\n".join(dependencies)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if requirements.txt differs from pyproject.toml",
    )
    args = parser.parse_args()
    expected = generated_requirements()
    if args.check:
        actual = REQUIREMENTS_FILE.read_text(encoding="utf-8")
        if actual != expected:
            parser.error(
                "requirements.txt is out of date; run "
                "`python scripts/sync_requirements.py`"
            )
        return 0
    REQUIREMENTS_FILE.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
