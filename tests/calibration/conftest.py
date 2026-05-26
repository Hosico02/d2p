"""Pytest collection guard for the calibration baseline tree.

The directories under tests/calibration/baselines/<kind>/<name>/ are
sample projects, not parts of the d2p test suite. Their internal
test files (e.g. flask-clean/tests/test_app.py) import from the
baseline's own root (`from app import app`), which only resolves
when the baseline directory itself is the project root — never when
collected from the parent d2p repo.

The calibration harness exercises these baselines via subprocess
(see d2p/agents/pre_evidence.py); they are never expected to be
collected by the parent pytest run."""
collect_ignore_glob = ["baselines/*/tests/*", "baselines/*/test_*"]
