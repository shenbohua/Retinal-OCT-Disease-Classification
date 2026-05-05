from __future__ import annotations

"""Deprecated compatibility module.

All deep-learning logic now lives under `src.deeplearning`.
This module re-exports `run_dl_experiment` for backward compatibility.
"""

from src.deeplearning.run_experiment import DLProtocolError, run_dl_experiment

__all__ = ["run_dl_experiment", "DLProtocolError"]
