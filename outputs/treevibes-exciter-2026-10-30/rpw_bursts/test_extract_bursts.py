"""Synthetic checks of timing/background rejection; not insect-identity tests."""
import unittest

import numpy as np
from extract_bursts import RATE, extract


class ExtractionTests(unittest.TestCase):
    def test_silence_and_stationary_background(self):
        self.assertEqual(extract(np.zeros(RATE * 10))[1], [])
        rng = np.random.default_rng(10)
        noise = rng.normal(0, 0.001, RATE * 10)
        self.assertEqual(extract(noise)[1], [])
        tone = 0.01 * np.sin(2 * np.pi * 500 * np.arange(RATE * 10) / RATE)
        self.assertEqual(extract(tone)[1], [])

    def test_known_burst_timing_and_preservation(self):
        rng = np.random.default_rng(20)
        background = rng.normal(0, 0.001, RATE * 10)
        x = background.copy()
        for onset in (2., 6.):
            n = int(0.08 * RATE)
            burst = 0.04 * np.hanning(n) * np.sin(2 * np.pi * 900 * np.arange(n) / RATE)
            x[int(onset * RATE):int(onset * RATE) + n] += burst
        y, events, stats = extract(x)
        self.assertEqual(len(events), 2)
        for onset, event in zip((2., 6.), events):
            self.assertLessEqual(event['start_sample'], onset * RATE)
            self.assertGreaterEqual(event['end_sample'], (onset + .08) * RATE)
            self.assertLess((event['end_sample'] - event['start_sample']) / RATE, .3)
        a, b = int(2.02 * RATE), int(2.06 * RATE)
        self.assertGreater(np.corrcoef(y[a:b], (x - background)[a:b])[0, 1], .95)
        self.assertLess(stats['background_output_rms'], stats['background_input_rms'] * .5)
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(len(y), len(x))


if __name__ == '__main__':
    unittest.main()
