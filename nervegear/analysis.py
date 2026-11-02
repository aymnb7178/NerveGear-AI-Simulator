"""Band-power analysis and a simple rule-based state classifier."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .signal import BANDS, FS


def band_powers(x: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """Absolute power per band from a Hann-windowed FFT periodogram."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("need a 1-D signal with at least 8 samples")
    x = x - x.mean()
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win)) ** 2 / (win ** 2).sum()
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(spec[m].sum())
    return out


def relative_powers(powers: Dict[str, float]) -> Dict[str, float]:
    total = sum(powers.values())
    if total <= 0:
        return {k: 0.0 for k in powers}
    return {k: v / total for k, v in powers.items()}


def engagement_index(powers: Dict[str, float]) -> float:
    """beta / (alpha + theta): a commonly used engagement ratio in EEG literature."""
    denom = powers["alpha"] + powers["theta"]
    return powers["beta"] / denom if denom > 0 else 0.0


def classify_state(powers: Dict[str, float]) -> str:
    """Rule-based label: 'focused', 'relaxed' or 'drowsy' (simulation only)."""
    rel = relative_powers(powers)
    if rel["beta"] >= rel["alpha"] and rel["beta"] >= rel["theta"] + rel["delta"] - 0.1 and rel["beta"] > 0.3:
        return "focused"
    if rel["delta"] + rel["theta"] > 0.5:
        return "drowsy"
    return "relaxed"
