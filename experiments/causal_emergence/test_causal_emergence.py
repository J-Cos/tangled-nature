#!/usr/bin/env python3
"""Unit tests for causal_emergence.py"""
import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from causal_emergence import (
    shannon_entropy, compute_ei, build_tpm,
    coarse_grain_micro, coarse_grain_meso1,
    coarse_grain_meso2, coarse_grain_macro,
)


class TestShannonEntropy(unittest.TestCase):
    def test_uniform(self):
        """Uniform dist over N states → log₂(N) bits."""
        self.assertAlmostEqual(shannon_entropy([0.25, 0.25, 0.25, 0.25]),
                                2.0, places=5)

    def test_deterministic(self):
        """Delta dist → 0 bits."""
        self.assertAlmostEqual(shannon_entropy([1.0, 0.0, 0.0]), 0.0)

    def test_empty(self):
        self.assertAlmostEqual(shannon_entropy([]), 0.0)

    def test_binary(self):
        self.assertAlmostEqual(shannon_entropy([0.5, 0.5]), 1.0, places=5)


class TestComputeEI(unittest.TestCase):
    def test_identity_tpm(self):
        """Identity TPM → maximum EI = log₂(N)."""
        N = 4
        tpm = np.eye(N)
        result = compute_ei(tpm)
        self.assertAlmostEqual(result["ei"], np.log2(N), places=5)
        self.assertAlmostEqual(result["determinism"], np.log2(N), places=5)
        self.assertAlmostEqual(result["degeneracy"], 0.0, places=5)

    def test_uniform_tpm(self):
        """Uniform TPM → EI = 0 (no causal info)."""
        N = 4
        tpm = np.ones((N, N)) / N
        result = compute_ei(tpm)
        self.assertAlmostEqual(result["ei"], 0.0, places=5)

    def test_single_state(self):
        """Single state → EI = 0."""
        result = compute_ei(np.array([[1.0]]))
        self.assertEqual(result["ei"], 0.0)

    def test_partial_determinism(self):
        """TPM with some randomness → 0 < EI < log₂(N)."""
        tpm = np.array([
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.2],
            [0.1, 0.0, 0.9],
        ])
        result = compute_ei(tpm)
        self.assertGreater(result["ei"], 0)
        self.assertLess(result["ei"], np.log2(3))
        self.assertGreater(result["determinism"], result["degeneracy"])


class TestBuildTPM(unittest.TestCase):
    def test_simple_sequence(self):
        """A → B → A → B → ... should yield clear transition probs."""
        states = ["A", "B", "A", "B", "A", "B"]
        tpm, labels = build_tpm(states)
        # A→B should be 1.0, B→A should be 1.0
        a_idx = labels.index("A")
        b_idx = labels.index("B")
        self.assertAlmostEqual(tpm[a_idx, b_idx], 1.0)
        self.assertAlmostEqual(tpm[b_idx, a_idx], 1.0)

    def test_single_state(self):
        tpm, labels = build_tpm(["X", "X", "X"])
        self.assertEqual(tpm.shape, (1, 1))
        self.assertAlmostEqual(tpm[0, 0], 1.0)

    def test_three_state_cycle(self):
        states = [1, 2, 3, 1, 2, 3, 1]
        tpm, labels = build_tpm(states)
        self.assertEqual(tpm.shape[0], 3)
        # Each state transitions to exactly one other
        for i in range(3):
            self.assertAlmostEqual(tpm[i].max(), 1.0)


class TestCoarseGraining(unittest.TestCase):
    def test_micro_empty(self):
        result = coarse_grain_micro({})
        self.assertEqual(len(result), 10)

    def test_micro_with_species(self):
        sp = {1: 100, 2: 50, 3: 5, 4: 1}
        result = coarse_grain_micro(sp, top_k=5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], 4)  # 100 → bin 4 (64+)

    def test_meso1(self):
        sp = {1: 100, 2: 50, 3: 10, 4: 5, 5: 1}
        result = coarse_grain_meso1(sp, n_rank_bins=5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], 4)  # top species holds majority

    def test_meso2(self):
        result = coarse_grain_meso2(5000, 50, n_range=(0, 10000), s_range=(0, 100))
        self.assertEqual(result, (2, 2))  # middle bins

    def test_macro(self):
        self.assertEqual(coarse_grain_macro(0, n_range=(0, 10000)), 0)
        self.assertEqual(coarse_grain_macro(9999, n_range=(0, 10000)), 4)
        self.assertEqual(coarse_grain_macro(5000, n_range=(0, 10000)), 2)


if __name__ == "__main__":
    unittest.main()
