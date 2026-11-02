"""Synthetic EEG-like signal generator.

Everything here is SIMULATED: a sum of sine waves in the classic EEG frequency
bands plus random noise. It is not recorded from a person and not real EEG.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

FS = 256  # sampling rate in Hz (typical for consumer EEG headsets)

BANDS: Dict[str, tuple] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Amplitude (arbitrary units) of each band per simulated mental state.
SCENARIOS: Dict[str, Dict[str, float]] = {
    "relaxed": {"delta": 4, "theta": 5, "alpha": 20, "beta": 5, "gamma": 2},
    "focused": {"delta": 3, "theta": 4, "alpha": 6, "beta": 18, "gamma": 5},
    "drowsy": {"delta": 15, "theta": 16, "alpha": 6, "beta": 3, "gamma": 1},
}


def generate(scenario: str, seconds: float = 4.0, fs: int = FS,
             noise: float = 2.0, seed: Optional[int] = None) -> np.ndarray:
    """Return ``seconds * fs`` samples for a scenario. Same seed -> same output."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    sig = np.zeros_like(t)
    for band, amp in SCENARIOS[scenario].items():
        lo, hi = BANDS[band]
        # three tones per band at random frequencies and phases
        for f in rng.uniform(lo + 0.2, hi - 0.2, size=3):
            sig += (amp / 3.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    sig += rng.normal(0.0, noise, size=t.shape)
    return sig
