"""Reproducibility: fix Python/NumPy RNG for a single run (sklearn uses explicit random_state elsewhere)."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set stdlib and NumPy RNGs. Always pair with ``random_state=`` on estimators."""
    random.seed(seed)
    np.random.seed(seed)
