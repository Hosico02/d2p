"""Tests for d2p._invariants and d2p._logging."""
from __future__ import annotations

import logging
import os
import pathlib

import pytest

from d2p._invariants import InvariantError, ensure, invariant, require
from d2p._logging import attach_run_log, configure


@pytest.fixture(autouse=True)
def _reset_invariants_env(monkeypatch):
    monkeypatch.delenv("D2P_INVARIANTS", raising=False)


class TestInvariants:
    def test_require_passes_when_true(self):
        require(True, "should not raise")

    def test_require_raises_in_strict_mode(self):
        with pytest.raises(InvariantError) as exc:
            require(False, "bad input", x=42)
        assert "bad input" in str(exc.value)
        assert "x=42" in str(exc.value)
        assert exc.value.kind == "require"
        assert exc.value.ctx == {"x": 42}

    def test_ensure_raises_with_distinct_kind(self):
        with pytest.raises(InvariantError) as exc:
            ensure(False, "post failed")
        assert exc.value.kind == "ensure"

    def test_invariant_raises_with_distinct_kind(self):
        with pytest.raises(InvariantError) as exc:
            invariant(False, "mid failed")
        assert exc.value.kind == "invariant"

    def test_invariant_error_subclasses_assertion_error(self):
        # legacy `except AssertionError` blocks still catch
        try:
            require(False, "x")
        except AssertionError as e:
            assert isinstance(e, InvariantError)
        else:
            pytest.fail("expected raise")

    def test_warn_mode_logs_but_does_not_raise(self, monkeypatch, caplog):
        monkeypatch.setenv("D2P_INVARIANTS", "warn")
        with caplog.at_level(logging.ERROR, logger="d2p.invariants"):
            require(False, "would have failed", n=7)
        assert any("would have failed" in r.message for r in caplog.records)
        assert any("n=7" in r.message for r in caplog.records)

    def test_off_mode_is_silent_and_does_not_raise(self, monkeypatch, caplog):
        monkeypatch.setenv("D2P_INVARIANTS", "off")
        with caplog.at_level(logging.ERROR, logger="d2p.invariants"):
            require(False, "ignored")
            ensure(False, "also ignored")
            invariant(False, "also ignored")
        assert caplog.records == []


class TestLogging:
    def test_configure_installs_stderr_handler(self):
        configure(verbose=False)
        root = logging.getLogger()
        kinds = [getattr(h, "_d2p_kind", None) for h in root.handlers]
        assert "stderr" in kinds

    def test_configure_is_idempotent_for_stderr(self):
        configure(verbose=False)
        configure(verbose=False)
        root = logging.getLogger()
        n = sum(1 for h in root.handlers
                if getattr(h, "_d2p_kind", None) == "stderr")
        assert n == 1

    def test_verbose_flag_lowers_stderr_level(self):
        configure(verbose=True)
        root = logging.getLogger()
        stderr = [h for h in root.handlers
                  if getattr(h, "_d2p_kind", None) == "stderr"][0]
        assert stderr.level == logging.DEBUG

    def test_env_log_level_overrides_verbose(self, monkeypatch):
        monkeypatch.setenv("D2P_LOG_LEVEL", "WARNING")
        configure(verbose=True)
        root = logging.getLogger()
        stderr = [h for h in root.handlers
                  if getattr(h, "_d2p_kind", None) == "stderr"][0]
        assert stderr.level == logging.WARNING

    def test_attach_run_log_writes_file(self, tmp_path: pathlib.Path):
        configure(verbose=False)
        run_dir = tmp_path / "run-X"
        attach_run_log(run_dir)
        log = logging.getLogger("d2p.test")
        log.info("hello world from test")
        # FileHandler flushes per-record by default; force just in case.
        for h in logging.getLogger().handlers:
            h.flush()
        assert (run_dir / "d2p.log").is_file()
        body = (run_dir / "d2p.log").read_text()
        assert "hello world from test" in body

    def test_attach_run_log_is_idempotent_per_dir(self, tmp_path: pathlib.Path):
        configure(verbose=False)
        run_dir = tmp_path / "run-Y"
        attach_run_log(run_dir)
        attach_run_log(run_dir)
        root = logging.getLogger()
        n = sum(1 for h in root.handlers
                if getattr(h, "_d2p_kind", None) == "run-file")
        # Includes only this run_dir's handler; other tests may have added
        # handlers for different run_dirs that are still attached. Filter.
        ours = [h for h in root.handlers
                if getattr(h, "_d2p_kind", None) == "run-file"
                and getattr(h, "baseFilename", "").endswith("run-Y/d2p.log")]
        assert len(ours) == 1
