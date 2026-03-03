#!/usr/bin/env python3
"""Unit tests for quake_analysis.py"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from quake_analysis import (
    jaccard_distance, compute_turnover, smooth, detect_quakes
)


class TestJaccardDistance(unittest.TestCase):
    def test_identical_sets(self):
        self.assertAlmostEqual(jaccard_distance({1, 2, 3}, {1, 2, 3}), 0.0)

    def test_disjoint_sets(self):
        self.assertAlmostEqual(jaccard_distance({1, 2}, {3, 4}), 1.0)

    def test_partial_overlap(self):
        # {1,2,3} ∩ {2,3,4} = {2,3}, union = {1,2,3,4}
        # Jaccard = 1 - 2/4 = 0.5
        self.assertAlmostEqual(jaccard_distance({1, 2, 3}, {2, 3, 4}), 0.5)

    def test_empty_sets(self):
        self.assertAlmostEqual(jaccard_distance(set(), set()), 0.0)

    def test_one_empty(self):
        self.assertAlmostEqual(jaccard_distance({1, 2}, set()), 1.0)

    def test_subset(self):
        # {1,2} ∩ {1,2,3} = {1,2}, union = {1,2,3}
        # Jaccard = 1 - 2/3 ≈ 0.333
        self.assertAlmostEqual(jaccard_distance({1, 2}, {1, 2, 3}), 1 / 3, places=5)


class TestComputeTurnover(unittest.TestCase):
    def test_no_change(self):
        """Identical species across gens should give 0 turnover."""
        snapshots = [
            {"species": {1: 10, 2: 5}},
            {"species": {1: 12, 2: 3}},
            {"species": {1: 8, 2: 7}},
        ]
        t = compute_turnover(snapshots)
        self.assertEqual(len(t), 3)
        self.assertAlmostEqual(t[0], 0.0)  # first gen
        self.assertAlmostEqual(t[1], 0.0)  # same species
        self.assertAlmostEqual(t[2], 0.0)  # same species

    def test_complete_turnover(self):
        """Completely different species should give turnover=1."""
        snapshots = [
            {"species": {1: 10, 2: 5}},
            {"species": {3: 10, 4: 5}},
        ]
        t = compute_turnover(snapshots)
        self.assertAlmostEqual(t[1], 1.0)

    def test_partial_turnover(self):
        snapshots = [
            {"species": {1: 10, 2: 5, 3: 3}},
            {"species": {2: 8, 3: 4, 4: 2}},
        ]
        # {1,2,3} ∩ {2,3,4} = {2,3}, union = {1,2,3,4}
        t = compute_turnover(snapshots)
        self.assertAlmostEqual(t[1], 0.5)


class TestSmooth(unittest.TestCase):
    def test_window_1(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(smooth(arr, 1), arr)

    def test_constant(self):
        arr = np.ones(20) * 5.0
        np.testing.assert_array_almost_equal(smooth(arr, 5), arr)

    def test_output_length(self):
        arr = np.random.randn(100)
        result = smooth(arr, 10)
        self.assertEqual(len(result), len(arr))


class TestDetectQuakes(unittest.TestCase):
    def test_no_quakes(self):
        """Constant low turnover should detect no quakes."""
        gens = np.arange(1000)
        turnover = np.ones(1000) * 0.01
        quakes = detect_quakes(turnover, gens, window=10, threshold_factor=2.0)
        self.assertEqual(len(quakes), 0)

    def test_single_spike(self):
        """A single spike should be detected as a quake."""
        gens = np.arange(1000)
        turnover = np.ones(1000) * 0.01
        turnover[400:420] = 0.5  # spike
        quakes = detect_quakes(turnover, gens, window=5, threshold_factor=2.0, min_gap=50)
        self.assertGreaterEqual(len(quakes), 1)
        # Peak should be in the spike region
        self.assertTrue(any(395 <= q[2] <= 425 for q in quakes))

    def test_two_spikes_far_apart(self):
        """Two well-separated spikes should be detected as two quakes."""
        gens = np.arange(2000)
        turnover = np.ones(2000) * 0.01
        turnover[300:320] = 0.5
        turnover[1500:1520] = 0.5
        quakes = detect_quakes(turnover, gens, window=5, threshold_factor=2.0, min_gap=50)
        self.assertEqual(len(quakes), 2)

    def test_merge_close_spikes(self):
        """Two close spikes should be merged into one quake."""
        gens = np.arange(1000)
        turnover = np.ones(1000) * 0.01
        turnover[300:310] = 0.5
        turnover[330:340] = 0.5
        quakes = detect_quakes(turnover, gens, window=5, threshold_factor=2.0, min_gap=100)
        self.assertEqual(len(quakes), 1)


if __name__ == "__main__":
    unittest.main()
