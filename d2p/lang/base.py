"""LanguageAdapter protocol — every language-specific behavior lives behind here."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..fs import Sandbox


@runtime_checkable
class LanguageAdapter(Protocol):
    """All methods are best-effort; returning empty/no-op means 'unsupported'."""

    name: str
    test_corpus_dir: str   # e.g. "tests/d2p_qa" — d2p stores generated tests here

    # ---- health ----
    def discover_modules(self, sandbox: Sandbox) -> list[str]:
        """Modules to probe for import-health (e.g. top-level Python files)."""
        ...

    def import_probe(self, sandbox: Sandbox,
                     modules: list[str]) -> dict[str, str]:
        """Return {module: 'ok' | <error-message>} — must be cheap and robust."""
        ...

    # ---- write-time safety ----
    def syntax_check(self, sandbox: Sandbox, rel_path: str) -> str:
        """Return non-empty error string if the file at rel_path has a syntax error."""
        ...

    # ---- QA / tests ----
    def test_template(self) -> str:
        """A skeleton the QA prompt shows to the model for new tests."""
        ...

    def test_path(self, slug: str) -> str:
        """Where to place a freshly-generated test file."""
        ...

    def test_runner_cmd(self, rel_path: str, *,
                        sandbox: Sandbox | None = None) -> list[str]:
        """Command to run a single test file. Empty list = unsupported.
        sandbox is optional; adapters may inspect file content to pick a runner.
        """
        ...


class NullAdapter:
    """No-op fallback for unknown / unsupported languages.

    d2p still functions with this adapter — it just loses health-rollback,
    syntax verification, and test-corpus generation. Feature execution and
    Analyzer/Planner remain fully functional.
    """

    name = "unknown"
    test_corpus_dir = "tests/d2p_qa"

    def discover_modules(self, sandbox: Sandbox) -> list[str]:
        return []

    def import_probe(self, sandbox: Sandbox, modules: list[str]) -> dict[str, str]:
        return {}

    def syntax_check(self, sandbox: Sandbox, rel_path: str) -> str:
        return ""

    def test_template(self) -> str:
        return ""

    def test_path(self, slug: str) -> str:
        return f"{self.test_corpus_dir}/{slug}"

    def test_runner_cmd(self, rel_path: str, *,
                        sandbox: Sandbox | None = None) -> list[str]:
        return []
