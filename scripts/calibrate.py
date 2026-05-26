#!/usr/bin/env python3
"""Thin CLI wrapper for the calibration harness.

Usage:
    python scripts/calibrate.py \\
        --baselines tests/calibration/baselines \\
        --model minimax-m2.7-hs \\
        --out tests/calibration/reports/$(date +%Y%m%d-%H%M%S)/

Exit codes:
    0  all baselines completed AND §11 single-pass criteria met
    1  all baselines completed but at least one criterion not met
    2  harness errored (baseline error, model misconfig, etc.)

See docs/superpowers/specs/2026-05-26-d2p-calibration-harness-design.md
in the demo2project repo for design rationale.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from d2p.calibration import main

if __name__ == "__main__":
    sys.exit(main())
