#!/usr/bin/env python3
"""Unit tests for preprocess.py"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import parse_jsonl, preprocess_ensemble


class TestParseJsonl(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_jsonl(self, lines, fname="test.jsonl"):
        path = os.path.join(self.tmpdir, fname)
        with open(path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return path

    def test_basic_snapshot(self):
        """Test parsing snapshot without species data."""
        path = self._write_jsonl([
            {"type": "snapshot", "gen": 100, "n": 500, "s": 10},
            {"type": "snapshot", "gen": 200, "n": 600, "s": 12},
        ])
        ts, sad = parse_jsonl(path, sim_id=1)
        self.assertEqual(len(ts), 2)
        self.assertEqual(len(sad), 0)
        self.assertEqual(ts[0]["gen"], 100)
        self.assertEqual(ts[0]["n"], 500)
        self.assertEqual(ts[0]["sim_id"], 1)

    def test_snapshot_with_species(self):
        """Test parsing snapshot with species abundances."""
        path = self._write_jsonl([
            {"type": "snapshot", "gen": 100, "n": 10, "s": 3,
             "species": [[0, 5], [1, 3], [2, 2]]},
        ])
        ts, sad = parse_jsonl(path, sim_id=2)
        self.assertEqual(len(ts), 1)
        self.assertEqual(len(sad), 3)
        self.assertEqual(sad[0]["genome"], 0)
        self.assertEqual(sad[0]["abundance"], 5)
        self.assertEqual(sad[2]["abundance"], 2)

    def test_qess_and_final(self):
        """Test parsing qess and final entries."""
        path = self._write_jsonl([
            {"type": "qess", "gen": 500, "n": 200, "s": 8, "cv": 0.03,
             "species": [[10, 100], [20, 100]]},
            {"type": "final", "gen": 500, "n": 200, "s": 8, "qess": True},
        ])
        ts, sad = parse_jsonl(path, sim_id=1)
        self.assertEqual(len(ts), 2)  # both qess and final
        self.assertEqual(len(sad), 2)  # species only from qess

    def test_empty_file(self):
        """Test parsing empty file."""
        path = self._write_jsonl([])
        ts, sad = parse_jsonl(path, sim_id=1)
        self.assertEqual(len(ts), 0)
        self.assertEqual(len(sad), 0)

    def test_non_json_lines_ignored(self):
        """Test that non-JSON lines are skipped."""
        path = os.path.join(self.tmpdir, "mixed.jsonl")
        with open(path, "w") as f:
            f.write("# comment\n")
            f.write("\n")
            f.write(json.dumps({"type": "snapshot", "gen": 1, "n": 10, "s": 2}) + "\n")
            f.write("not json\n")
        ts, sad = parse_jsonl(path, sim_id=1)
        self.assertEqual(len(ts), 1)


class TestPreprocessEnsemble(unittest.TestCase):
    def test_ensemble_processing(self):
        """Test processing multiple sim files."""
        tmpdir = tempfile.mkdtemp()
        data_dir = os.path.join(tmpdir, "data")
        out_dir = os.path.join(tmpdir, "results")
        os.makedirs(data_dir)

        for i in range(1, 4):
            path = os.path.join(data_dir, f"sim_{i:02d}.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({
                    "type": "snapshot", "gen": 100, "n": 100 * i, "s": 5 * i,
                    "species": [[g, 10] for g in range(5 * i)]
                }) + "\n")

        result = preprocess_ensemble(data_dir, out_dir)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "timeseries.csv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "sad_snapshots.csv")))

    def test_no_files(self):
        """Test with empty data directory."""
        tmpdir = tempfile.mkdtemp()
        data_dir = os.path.join(tmpdir, "empty")
        os.makedirs(data_dir)
        result = preprocess_ensemble(data_dir, os.path.join(tmpdir, "out"))
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
