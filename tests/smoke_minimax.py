"""Live smoke test against api.minimax.io. Confirms auth + model before agents run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.llm import MiniMaxClient


def main() -> int:
    client = MiniMaxClient()
    text = client.chat(
        system="You are a smoke test. Reply with the single word PONG and nothing else.",
        user="ping",
        temperature=0.0,
    )
    print("RAW:", repr(text))
    ok = "PONG" in text.upper()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
