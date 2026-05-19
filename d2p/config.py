import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(Path.cwd() / ".env")
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.environ.get("MINIMAX_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"
        )
    )
    model: str = field(default_factory=lambda: os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"))
    max_iterations: int = 3
    parallel_executors: int = 4
    request_timeout: int = 240
    # Re-run the Analyzer every N iterations (0 = never, only at iter 1).
    # Use case: long runs where the project drifts away from the initial
    # feature plan; re-analysis refreshes the feature list while ESSENCE
    # and AUDIENCE remain immutable invariants.
    reanalyze_every: int = 0
    # Mark a QA bug as `wontfix` after this many failed fix attempts so the
    # orchestrator stops dispatching the same broken fix forever. The test
    # stays in the corpus — if it ever turns green later, it gets flipped
    # back to "fixed" automatically.
    qa_wontfix_after_attempts: int = 3

    def require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is not set. Put it in environment or in d2p/.env"
            )
