"""Render a PNG dashboard with OpenCV (no matplotlib needed)."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .signal import BANDS

COLORS = {"delta": (200, 120, 60), "theta": (200, 200, 60), "alpha": (80, 200, 80),
          "beta": (60, 140, 240), "gamma": (180, 80, 220)}
STATE_COLOR = {"relaxed": (80, 200, 80), "focused": (60, 140, 240), "drowsy": (200, 120, 60)}


def render(signal: np.ndarray, records: List[dict], path: str) -> None:
    """``records``: one dict per window with keys state, engagement, rel (band->fraction)."""
    W, H = 1000, 620
    img = np.full((H, W, 3), 24, np.uint8)
    cv2.putText(img, "NerveGear AI Simulator  -  SYNTHETIC DATA, NOT REAL EEG", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # signal trace
    cv2.putText(img, "Simulated signal", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
    top, bot = 70, 230
    sig = np.asarray(signal, float)
    scale = (bot - top) / 2 / max(np.abs(sig).max(), 1e-9)
    xs = np.linspace(20, W - 20, sig.size)
    ys = (top + bot) / 2 - sig * scale
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, (120, 220, 255), 1, cv2.LINE_AA)

    # relative band power per window (stacked bars)
    cv2.putText(img, "Relative band power per window", (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
    n = len(records)
    bw = (W - 40) / max(n, 1)
    for i, r in enumerate(records):
        x0 = int(20 + i * bw)
        y = 280 + 200
        for band in BANDS:
            h = int(r["rel"][band] * 200)
            cv2.rectangle(img, (x0 + 1, y - h), (int(x0 + bw) - 1, y), COLORS[band], -1)
            y -= h
        cv2.rectangle(img, (x0 + 1, 490), (int(x0 + bw) - 1, 505), STATE_COLOR[r["state"]], -1)
        cv2.putText(img, f"{r['engagement']:.1f}", (x0 + 2, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
    cv2.putText(img, "state (colour) / engagement index (number)", (20, 548),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)

    # legend
    x = 20
    for band, c in COLORS.items():
        cv2.rectangle(img, (x, 580), (x + 14, 594), c, -1)
        cv2.putText(img, band, (x + 20, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        x += 100
    x += 20
    for s, c in STATE_COLOR.items():
        cv2.rectangle(img, (x, 580), (x + 14, 594), c, -1)
        cv2.putText(img, s, (x + 20, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        x += 110
    if not cv2.imwrite(path, img):
        raise OSError(f"could not write {path}")
