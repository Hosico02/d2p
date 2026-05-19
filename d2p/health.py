"""Lightweight import probe — distinguishes 'file parses but breaks runtime imports'
from 'file parses and runs cleanly'.

Language-agnostic: delegates to a `LanguageAdapter` (Python/JS/etc) for the
actual probe. NullAdapter degrades to a no-op safely.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .fs import Sandbox
from .lang import LanguageAdapter, adapter_for, detect_primary_language


class ProjectHealth:
    def __init__(self, sandbox: Sandbox,
                 adapter: Optional[LanguageAdapter] = None) -> None:
        self.sandbox = sandbox
        self.adapter = adapter or adapter_for(detect_primary_language(sandbox))

    def probe(self, modules: Iterable[str]) -> dict[str, str]:
        return self.adapter.import_probe(self.sandbox, list(modules))

    def default_modules(self) -> list[str]:
        return self.adapter.discover_modules(self.sandbox)
