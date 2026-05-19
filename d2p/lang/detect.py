"""Detect a project's primary language from its file inventory."""
from __future__ import annotations

from collections import Counter

from ..fs import Sandbox


_EXT_TO_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
}


def detect_primary_language(sandbox: Sandbox) -> str:
    """Return the dominant source language by file count.
    Returns 'unknown' if no recognised source files are present.
    """
    counts: Counter[str] = Counter()
    for p in sandbox.listing(max_entries=2000):
        if p.startswith("..."):
            continue
        for ext, lang in _EXT_TO_LANG.items():
            if p.endswith(ext):
                counts[lang] += 1
                break
    if not counts:
        return "unknown"
    # ties: prefer python, then typescript, then javascript, then go, then rust
    priority = {"python": 5, "typescript": 4, "javascript": 3, "go": 2, "rust": 1}
    return max(counts.items(), key=lambda kv: (kv[1], priority.get(kv[0], 0)))[0]
