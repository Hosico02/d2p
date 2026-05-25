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
    # Set of roles for which race-mode is enabled. When a role is in this
    # set AND a fallback model is configured for it, the orchestrator
    # runs primary.prepare() and fallback.prepare() in PARALLEL on every
    # task of that role. The first side to finish gets its commit tried
    # immediately; if commit succeeds the slow side is abandoned.
    #
    # Per-role design: race is a "2× LLM tokens for less wall time" trade.
    # It pays off most on `fix` tasks (single-task latency is high; primary
    # often fails on hard bugs anyway). It's a worse trade on `executor`
    # because feature tasks usually succeed on primary, so race just
    # doubles cost. Default = empty (race off everywhere).
    #
    # When race is active for a role, Executor.commit() is invoked with
    # max_fix_attempts=1 so we don't compound race × MAX_FIX_ATTEMPTS
    # (the design bug from the original opt-in `--fix-race`). The race
    # itself IS the retry.
    #
    # CLI accepts: --race-mode (no arg = all), --race-mode fix,
    # --race-mode executor, --race-mode fix,executor, --race-mode none.
    race_roles: set[str] = field(default_factory=set)
    hub_url: str | None = field(default_factory=lambda: os.environ.get("HUB_URL"))
    hub_token: str | None = field(default_factory=lambda: os.environ.get("HUB_TOKEN"))
    # Adversarial verifier (an independent Opus-class call after each iter's
    # QA fix sweep). Runs the fixed-point convergence state machine — loop
    # terminates after 2 consecutive `no_new_findings` passes, OR on a
    # blocking `fail`, OR when max_iterations hits. Opt-in via env to keep
    # existing runs unchanged until calibration shows verify is worth its
    # +25% per-iter cost. See
    # ../demo2project/docs/superpowers/specs/2026-05-22-d2p-verify-agent-design.md
    verify_enabled: bool = field(default_factory=lambda: os.environ.get(
        "D2P_VERIFY_ENABLED", "").lower() in {"1", "true", "yes", "on"})
    # Streak length required for clean convergence. Default 2 per the spec
    # (1 is noise-vulnerable, 3 is overkill).
    verify_streak_required: int = 2

    def require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is not set. Put it in environment or in d2p/.env"
            )
