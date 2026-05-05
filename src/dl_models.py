from __future__ import annotations

"""Deprecated compatibility module.

All deep-learning model/training components now live under `src.deeplearning`.
This module re-exports `DLRunConfig` for backward compatibility.
"""

from src.deeplearning.trainer import DLRunConfig

__all__ = ["DLRunConfig"]
