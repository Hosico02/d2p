"""Sandboxed filesystem helpers — every path is forced under the project root."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class Sandbox:
    MAX_BYTES = 200_000  # per-file safety cap for reads we send to the LLM

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"sandbox root not a directory: {self.root}")

    def _resolve(self, relative: str) -> Path:
        p = (self.root / relative).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"path escapes sandbox: {relative}")
        return p

    def listing(self, max_entries: int = 200) -> list[str]:
        skip = {".git", "__pycache__", "node_modules", ".venv", "venv",
                ".pytest_cache", ".mypy_cache", "dist", "build", ".d2p"}
        out: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if any(part in skip for part in path.relative_to(self.root).parts):
                continue
            if path.is_dir():
                continue
            out.append(str(path.relative_to(self.root)))
            if len(out) >= max_entries:
                out.append("... (truncated)")
                break
        return out

    def read(self, relative: str) -> str:
        p = self._resolve(relative)
        if not p.is_file():
            return ""
        data = p.read_bytes()
        if len(data) > self.MAX_BYTES:
            data = data[: self.MAX_BYTES] + b"\n... (truncated)"
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary file, {len(data)} bytes>"

    def write(self, relative: str, content: str) -> str:
        p = self._resolve(relative)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return str(p.relative_to(self.root))

    def delete(self, relative: str) -> bool:
        p = self._resolve(relative)
        if p.is_file():
            p.unlink()
            return True
        return False

    # ---- snapshot / restore for rollback on regression ----------------------

    _MISSING = object()

    def snapshot(self, paths: list[str]) -> dict[str, Any]:
        """Capture current contents of `paths`. Missing files mark `_MISSING`."""
        snap: dict[str, Any] = {}
        for rel in paths:
            p = self._resolve(rel)
            if p.is_file():
                snap[rel] = p.read_bytes()
            else:
                snap[rel] = self._MISSING
        return snap

    def restore(self, snapshot: dict[str, Any]) -> list[str]:
        """Restore previously-captured snapshot. Returns the list of paths restored."""
        restored: list[str] = []
        for rel, data in snapshot.items():
            p = self._resolve(rel)
            if data is self._MISSING:
                if p.is_file():
                    p.unlink()
                    restored.append(rel)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                restored.append(rel)
        return restored
