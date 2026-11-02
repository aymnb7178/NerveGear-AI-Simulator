import os
import tempfile
import unittest

import numpy as np

from nervegear.analysis import (band_powers, classify_state, engagement_index,
                                relative_powers)
from nervegear.signal import FS, SCENARIOS, generate
from nervegear.simulate import accuracy, run


class TestSignal(unittest.TestCase):
    def test_length_and_determinism(self):
        a = generate("relaxed", seconds=2, seed=5)
        b = generate("relaxed", seconds=2, seed=5)
        self.assertEqual(a.size, 2 * FS)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, generate("relaxed", seconds=2, seed=6)))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            generate("nope")
        with self.assertRaises(ValueError):
            generate("focused", seconds=0)


class TestAnalysis(unittest.TestCase):
    def test_pure_10hz_sine_is_alpha(self):
        t = np.arange(4 * FS) / FS
        p = band_powers(np.sin(2 * np.pi * 10 * t))
        self.assertEqual(max(p, key=p.get), "alpha")

    def test_pure_20hz_sine_is_beta(self):
        t = np.arange(4 * FS) / FS
        p = band_powers(np.sin(2 * np.pi * 20 * t))
        self.assertEqual(max(p, key=p.get), "beta")

    def test_relative_powers_sum_to_one(self):
        rel = relative_powers(band_powers(generate("focused", seed=1)))
        self.assertAlmostEqual(sum(rel.values()), 1.0, places=9)

    def test_engagement_ordering(self):
        e = {s: engagement_index(band_powers(generate(s, seed=3))) for s in SCENARIOS}
        self.assertGreater(e["focused"], e["relaxed"])
        self.assertGreater(e["focused"], e["drowsy"])

    def test_each_scenario_is_recognised_across_seeds(self):
        for s in SCENARIOS:
            for seed in range(10):
                with self.subTest(s=s, seed=seed):
                    self.assertEqual(classify_state(band_powers(generate(s, seed=seed))), s)

    def test_bad_signal(self):
        with self.assertRaises(ValueError):
            band_powers(np.zeros(3))
        self.assertEqual(engagement_index({"alpha": 0, "theta": 0, "beta": 1}), 0.0)


class TestSimulate(unittest.TestCase):
    def test_run_shapes_and_accuracy(self):
        sig, rec = run(["relaxed", "focused", "drowsy"], seed=1)
        self.assertEqual(len(rec), 9)
        self.assertEqual(sig.size, 3 * 6 * FS)
        self.assertGreaterEqual(accuracy(rec), 0.9)

    def test_dashboard_png(self):
        try:
            from nervegear.dashboard import render
        except ImportError:
            self.skipTest("opencv not installed")
        sig, rec = run(["relaxed", "focused"], seed=2)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "d.png")
            render(sig, rec, p)
            self.assertGreater(os.path.getsize(p), 1000)


if __name__ == "__main__":
    unittest.main()
