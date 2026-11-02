"""Run a scenario sequence, analyse it in 2-second windows, print a table, write CSV/PNG.

    python -m nervegear.simulate --scenarios relaxed focused drowsy --seed 1 --out out
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import List, Sequence

import numpy as np

from .analysis import band_powers, classify_state, engagement_index, relative_powers
from .signal import BANDS, FS, generate


def run(scenarios: Sequence[str], seconds_each: float = 6.0, window_s: float = 2.0,
        seed: int = 0):
    """Return (signal, records). One record per window with the TRUE scenario for comparison."""
    parts, truth = [], []
    for k, name in enumerate(scenarios):
        seg = generate(name, seconds=seconds_each, seed=seed + k)
        parts.append(seg)
        truth += [name] * int(seconds_each / window_s)
    signal = np.concatenate(parts)
    w = int(window_s * FS)
    records: List[dict] = []
    for i, true_state in enumerate(truth):
        pw = band_powers(signal[i * w:(i + 1) * w])
        records.append({
            "window": i, "true": true_state, "state": classify_state(pw),
            "engagement": engagement_index(pw), "rel": relative_powers(pw),
        })
    return signal, records


def accuracy(records: Sequence[dict]) -> float:
    """Fraction of windows where the rule matches the simulated scenario (synthetic only)."""
    return sum(r["state"] == r["true"] for r in records) / len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=["relaxed", "focused", "drowsy"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    signal, rec = run(a.scenarios, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    print(f"{'win':>3} {'simulated':>10} {'detected':>9} {'engagement':>11}")
    for r in rec:
        print(f"{r['window']:>3} {r['true']:>10} {r['state']:>9} {r['engagement']:>11.2f}")
    print(f"\nRule matches the simulated state in {accuracy(rec):.0%} of windows "
          "(synthetic data - says nothing about real EEG).")

    with open(os.path.join(a.out, "windows.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window", "simulated", "detected", "engagement"] + list(BANDS))
        for r in rec:
            wr.writerow([r["window"], r["true"], r["state"], f"{r['engagement']:.3f}"] +
                        [f"{r['rel'][b]:.3f}" for b in BANDS])
    try:
        from .dashboard import render
        render(signal, rec, os.path.join(a.out, "dashboard.png"))
        print(f"Wrote {a.out}/windows.csv and {a.out}/dashboard.png")
    except ImportError:
        print(f"Wrote {a.out}/windows.csv (install opencv-python for the PNG dashboard)")


if __name__ == "__main__":
    main()
