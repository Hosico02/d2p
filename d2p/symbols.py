"""Cheap, language-agnostic symbol extractor for Planner context.

We don't run language parsers — a few regex per language is enough to give the
Planner a map of what's already defined. Better imprecise + cheap than wrong.
"""
from __future__ import annotations

import re
from typing import Iterable

_RULES: dict[str, list[re.Pattern[str]]] = {
    ".py": [
        re.compile(r"^\s*class\s+([A-Z][\w]*)", re.MULTILINE),
        re.compile(r"^\s*def\s+([a-zA-Z_][\w]*)\s*\(", re.MULTILINE),
        re.compile(r"^\s*@\w+\.route\(['\"]([^'\"]+)['\"]", re.MULTILINE),
    ],
    ".js": [
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][\w]*)", re.MULTILINE),
        re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][\w]*)", re.MULTILINE),
    ],
    ".ts": [
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][\w]*)", re.MULTILINE),
        re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][\w]*)", re.MULTILINE),
    ],
    ".go": [re.compile(r"^\s*func\s+(?:\([^)]*\)\s+)?([A-Za-z_][\w]*)", re.MULTILINE)],
    ".rs": [
        re.compile(r"^\s*(?:pub\s+)?fn\s+([a-z_][\w]*)", re.MULTILINE),
        re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Z][\w]*)", re.MULTILINE),
    ],
}
_RULES[".jsx"] = _RULES[".js"]
_RULES[".tsx"] = _RULES[".ts"]


def extract_symbols(rel_path: str, content: str, max_per_file: int = 40) -> list[str]:
    if not content:
        return []
    ext = "." + rel_path.rsplit(".", 1)[-1].lower() if "." in rel_path else ""
    rules = _RULES.get(ext)
    if not rules:
        return []
    out: list[str] = []
    for rx in rules:
        for m in rx.finditer(content):
            sym = m.group(1)
            if sym not in out:
                out.append(sym)
                if len(out) >= max_per_file:
                    return out
    return out


def build_symbol_map(read_file, files: Iterable[str]) -> dict[str, list[str]]:
    """`read_file(path) -> str` is the reader (e.g. sandbox.read)."""
    out: dict[str, list[str]] = {}
    for path in files:
        syms = extract_symbols(path, read_file(path))
        if syms:
            out[path] = syms
    return out
