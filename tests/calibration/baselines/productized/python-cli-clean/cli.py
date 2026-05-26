"""Greeter CLI with argparse + --help."""
from __future__ import annotations

import argparse
import sys


def build_message(name: str) -> str:
    return f"Hello, {name}!"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="greet", description="Greet someone by name.")
    parser.add_argument("name", help="who to greet")
    args = parser.parse_args(argv)
    print(build_message(args.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
