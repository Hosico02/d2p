"""Greeter CLI."""
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: cli.py NAME", file=sys.stderr)
        return 1
    print(f"Hello, {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
