"""Read-only links to the developmental project and RoBoWoMo harness."""
from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALTECH_ROOT = PROJECT_ROOT.parents[1]
NEURODEV_ROOT = Path(os.environ.get(
    "NEURODEV_WM", CALTECH_ROOT / "RESEARCH" / "neurodev_wm")).expanduser()
ROBOWOMO_ROOT = Path(os.environ.get(
    "ROBOWOMO_ROOT", CALTECH_ROOT / "robowomo-attractors")).expanduser()

for required in (NEURODEV_ROOT / "runs" / "dev",
                 ROBOWOMO_ROOT / "src"):
    if not required.exists():
        raise ImportError(f"required source path not found: {required}")

# RoBoWoMo is inserted first so modules imported by rl_navigation resolve to
# the canonical harness. Nothing is copied into this extension.
for source in (ROBOWOMO_ROOT / "src", NEURODEV_ROOT / "src"):
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

import models  # noqa: E402
import rl_navigation  # noqa: E402
import stat_tests  # noqa: E402


DEV_RUNS = NEURODEV_ROOT / "runs" / "dev"
DEVELOPMENT_JSON = NEURODEV_ROOT / "runs" / "development.json"

__all__ = ["models", "rl_navigation", "stat_tests", "DEV_RUNS",
           "DEVELOPMENT_JSON", "NEURODEV_ROOT", "ROBOWOMO_ROOT"]

