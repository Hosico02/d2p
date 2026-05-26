"""Centralized logging configuration for d2p.

One call to `configure(verbose, run_dir)` from the CLI entry point sets up
both the stderr handler (INFO/DEBUG, human-readable) and a per-run file
handler at DEBUG level. Module loggers (`d2p.<name>`) propagate up to the
root logger configured here.

Env overrides:
  D2P_LOG_LEVEL  — stderr threshold (DEBUG/INFO/WARNING/ERROR). Wins over
                   the --verbose flag when set.
  D2P_LOG_FILE   — extra file path to mirror DEBUG output to. Useful for
                   tailing across runs without digging into <run_dir>.
"""
from __future__ import annotations

import logging
import os
import pathlib

_STDERR_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d | %(message)s"
_DATEFMT = "%H:%M:%S"

# Track installed file handlers so we don't double-attach on re-entry
# (orchestrator-driven resume can re-call configure() with a new run_dir).
_RUN_FILE_HANDLERS: dict[str, logging.FileHandler] = {}


def _resolve_stderr_level(verbose: bool) -> int:
    env = os.environ.get("D2P_LOG_LEVEL", "").strip().upper()
    if env:
        return getattr(logging, env, logging.INFO)
    return logging.DEBUG if verbose else logging.INFO


def configure(verbose: bool = False) -> None:
    """Install the stderr handler on the root logger.

    Called once from the CLI entry point. Safe to call again — replaces
    the existing stderr handler so a later `--verbose` toggle is honored.
    The optional `D2P_LOG_FILE` env mirror is also installed here.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers filter; root passes everything

    stderr_level = _resolve_stderr_level(verbose)
    existing_stderr = [h for h in root.handlers
                       if getattr(h, "_d2p_kind", None) == "stderr"]
    for h in existing_stderr:
        root.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setLevel(stderr_level)
    sh.setFormatter(logging.Formatter(_STDERR_FORMAT, datefmt=_DATEFMT))
    sh._d2p_kind = "stderr"  # type: ignore[attr-defined]
    root.addHandler(sh)

    extra = os.environ.get("D2P_LOG_FILE", "").strip()
    if extra:
        already = any(getattr(h, "_d2p_kind", None) == "env-file"
                      for h in root.handlers)
        if not already:
            path = pathlib.Path(extra).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATEFMT))
            fh._d2p_kind = "env-file"  # type: ignore[attr-defined]
            root.addHandler(fh)


def attach_run_log(run_dir: pathlib.Path) -> None:
    """Attach a DEBUG-level file handler at `<run_dir>/d2p.log`.

    Idempotent on the same run_dir. Calling with a different run_dir adds
    a second file sink rather than replacing — useful for `--resume`
    chains where each run dir gets its own log file.
    """
    root = logging.getLogger()
    key = str(run_dir.resolve())
    if key in _RUN_FILE_HANDLERS:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "d2p.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATEFMT))
    fh._d2p_kind = "run-file"  # type: ignore[attr-defined]
    root.addHandler(fh)
    _RUN_FILE_HANDLERS[key] = fh
