#!/usr/bin/env python3
"""Unit tests for orchestrator metric computations (METE D_KL, λ₂)."""

import math
import unittest
import numpy as np
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from orchestrator import compute_mete_dkl, compute_lambda2


class TestMETEDKL(unittest.TestCase):

    def test_dkl_positive(self):
        """D_KL should be ≥ 0 for any SAD."""
        abundances = [100, 50, 20, 10, 5, 3, 1, 1]
        dkl = compute_mete_dkl(abundances)
        self.assertGreaterEqual(dkl, 0.0, f"D_KL should be ≥ 0, got {dkl}")
        self.assertTrue(math.isfinite(dkl), f"D_KL should be finite, got {dkl}")

    def test_dkl_uniform_high(self):
        """Uniform SAD should have high D_KL (far from log-series)."""
        abundances = [10] * 50  # all species same abundance
        dkl = compute_mete_dkl(abundances)
        self.assertGreater(dkl, 0.1, f"Uniform SAD should deviate from METE, got {dkl}")

    def test_dkl_single_species(self):
        """Single species → D_KL = 0 (degenerate case)."""
        dkl = compute_mete_dkl([100])
        self.assertEqual(dkl, 0.0)

    def test_dkl_classic_logseries_shape(self):
        """A rough log-series-like SAD should have lower D_KL."""
        # Many rare species, few common ones
        logseries_like = [1]*50 + [2]*20 + [5]*10 + [10]*5 + [50]*2 + [200]
        uniform = [10] * 88
        dkl_log = compute_mete_dkl(logseries_like)
        dkl_uni = compute_mete_dkl(uniform)
        self.assertLess(dkl_log, dkl_uni,
                        f"Log-series SAD should have lower D_KL ({dkl_log}) "
                        f"than uniform ({dkl_uni})")

    def test_dkl_empty(self):
        """Empty or single-element lists should return 0."""
        self.assertEqual(compute_mete_dkl([]), 0.0)
        self.assertEqual(compute_mete_dkl([5]), 0.0)


class TestLambda2(unittest.TestCase):

    def test_complete_graph(self):
        """Complete graph of S nodes: λ₂ = S/(S-1)."""
        S = 5
        # J matrix where all off-diagonal entries are non-zero
        J = np.ones((S, S)) * 0.5
        np.fill_diagonal(J, 0.0)
        lam2 = compute_lambda2(J.tolist())
        expected = S / (S - 1)
        self.assertAlmostEqual(lam2, expected, places=5,
                               msg=f"Complete graph λ₂ should be {expected}, got {lam2}")

    def test_disconnected_graph(self):
        """Disconnected graph: λ₂ = 0."""
        # Diagonal matrix (no interactions)
        J = np.zeros((5, 5))
        lam2 = compute_lambda2(J.tolist())
        self.assertEqual(lam2, 0.0)

    def test_two_nodes_connected(self):
        """Two connected nodes: λ₂ = 2 (normalized Laplacian)."""
        J = [[0.0, 0.5], [0.3, 0.0]]
        lam2 = compute_lambda2(J)
        self.assertAlmostEqual(lam2, 2.0, places=5,
                               msg=f"Two connected nodes λ₂ should be 2.0, got {lam2}")

    def test_single_node(self):
        """Single node: λ₂ = 0."""
        lam2 = compute_lambda2([[0.0]])
        self.assertEqual(lam2, 0.0)

    def test_lambda2_range(self):
        """λ₂ of normalized Laplacian ∈ [0, 2]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            S = rng.integers(3, 20)
            J = rng.uniform(-1, 1, (S, S))
            np.fill_diagonal(J, 0)
            lam2 = compute_lambda2(J.tolist())
            self.assertGreaterEqual(lam2, 0.0 - 1e-10)
            self.assertLessEqual(lam2, 2.0 + 1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
