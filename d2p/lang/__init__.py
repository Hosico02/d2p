"""Language adapters — keep d2p's safety nets and QA loop language-agnostic.

The orchestrator picks one `LanguageAdapter` per target project at startup
(via `detect_primary_language`). All language-specific knowledge (syntax
check, import probe, test framework, test runner) lives behind the adapter
protocol — d2p's core has no `.py`/`.js`/etc branches.
"""
from .base import LanguageAdapter, NullAdapter
from .detect import detect_primary_language
from .python import PythonAdapter
from .javascript import JSAdapter


def adapter_for(language: str) -> LanguageAdapter:
    """Return the adapter matching the detected primary language."""
    table: dict[str, LanguageAdapter] = {
        "python": PythonAdapter(),
        "javascript": JSAdapter(),
        "typescript": JSAdapter(),  # TS uses the same node toolchain
    }
    return table.get(language, NullAdapter())


__all__ = [
    "LanguageAdapter", "NullAdapter",
    "PythonAdapter", "JSAdapter",
    "detect_primary_language", "adapter_for",
]
