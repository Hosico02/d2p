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
    # Cap how many QA-fix tasks run in a single iteration (0 = no cap).
    # When the model+escalation cost per fix is high, this prevents one
    # iter from blowing the budget on N parallel sonnet retries. Bugs
    # left behind get picked up next iter — naturally favours bugs with
    # the lowest attempts (i.e. freshest to try, not the ones already
    # circling the drain).
    max_concurrent_fixes: int = 0
    # When True AND a fallback model is configured for the fix role, the
    # orchestrator runs primary.prepare() and fallback.prepare() in
    # PARALLEL on every fix task. If the primary commit fails, the
    # fallback's prepared LLM output is already available — skip the
    # sequential ~80s escalation wait. Costs 2× fix LLM calls per task
    # even when primary succeeds; only worth turning on when haiku fix
    # success rate is < 70% on the target codebase.
    fix_race: bool = False

    def require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is not set. Put it in environment or in d2p/.env"
            )
