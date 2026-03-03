#!/usr/bin/env python3
"""Unit tests for analyze.py"""
import json, os, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze import parse_fork_output, compute_fork_metrics, extract_top_species


class TestParseForkOutput(unittest.TestCase):
    def _write(self, lines):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for l in lines:
            f.write(json.dumps(l) + "\n")
        f.close()
        return f.name

    def test_basic(self):
        path = self._write([
            {"type": "snapshot", "gen": 100, "n": 500, "s": 10},
            {"type": "snapshot", "gen": 200, "n": 400, "s": 8},
        ])
        rows = parse_fork_output(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["n"], 500)

    def test_empty(self):
        path = self._write([])
        self.assertEqual(len(parse_fork_output(path)), 0)


class TestComputeForkMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        rows = [
            {"gen": i, "n": 1000, "s": 20} for i in range(200)
        ] + [
            {"gen": 200 + i, "n": 800, "s": 15} for i in range(200)
        ]
        m = compute_fork_metrics(rows, harvest_after=200)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["mean_n_baseline"], 1000)
        self.assertAlmostEqual(m["mean_n_harvest"], 800)
        self.assertAlmostEqual(m["delta_n"], -200)
        self.assertAlmostEqual(m["pct_n_change"], -20.0)

    def test_too_few_rows(self):
        self.assertIsNone(compute_fork_metrics([{"gen": 1, "n": 10, "s": 2}]))


class TestExtractTopSpecies(unittest.TestCase):
    def test_basic(self):
        tmpf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"species": {"10": 100, "20": 50, "30": 200, "40": 25}}, tmpf)
        tmpf.close()
        top = extract_top_species(tmpf.name, n_species=3)
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0][0], 30)  # highest abundance
        self.assertEqual(top[0][1], 200)
        self.assertEqual(top[1][0], 10)  # second


if __name__ == "__main__":
    unittest.main()
