#!/usr/bin/env python3
"""
Preprocess TNM JSONL output for METE analysis.

Extracts:
1. Time series CSV (gen, n, s, sim_id)  — from all snapshots
2. SAD snapshots CSV (gen, sim_id, genome, abundance) — species-level data
"""

import json
import csv
import sys
import os
import glob
from pathlib import Path


def parse_jsonl(fname, sim_id):
    """Parse a single JSONL file and return (timeseries_rows, sad_rows)."""
    ts_rows = []
    sad_rows = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("type") not in ("snapshot", "qess", "final"):
                continue
            gen = d.get("gen", 0)
            n = d.get("n", 0)
            s = d.get("s", 0)
            ts_rows.append({"gen": gen, "n": n, "s": s, "sim_id": sim_id})

            # Extract species abundances if present
            species = d.get("species")
            if species:
                for genome, abundance in species:
                    sad_rows.append({
                        "gen": gen,
                        "sim_id": sim_id,
                        "genome": genome,
                        "abundance": abundance,
                    })
    return ts_rows, sad_rows


def preprocess_ensemble(data_dir, output_dir):
    """Process all sim_*.jsonl files in data_dir."""
    os.makedirs(output_dir, exist_ok=True)

    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "sim_*.jsonl")))
    if not jsonl_files:
        print(f"No sim_*.jsonl files found in {data_dir}")
        return False

    all_ts = []
    all_sad = []

    for fname in jsonl_files:
        # Extract sim_id from filename (e.g., sim_01.jsonl -> 1)
        basename = Path(fname).stem
        sim_id = int(basename.split("_")[1])
        print(f"  Processing {basename}...", end=" ", flush=True)

        ts_rows, sad_rows = parse_jsonl(fname, sim_id)
        all_ts.extend(ts_rows)
        all_sad.extend(sad_rows)
        print(f"{len(ts_rows)} snapshots, {len(sad_rows)} species records")

    # Write time series CSV
    ts_path = os.path.join(output_dir, "timeseries.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gen", "n", "s", "sim_id"])
        w.writeheader()
        w.writerows(all_ts)
    print(f"  → Wrote {len(all_ts)} rows to {ts_path}")

    # Write SAD snapshots CSV
    sad_path = os.path.join(output_dir, "sad_snapshots.csv")
    with open(sad_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gen", "sim_id", "genome", "abundance"])
        w.writeheader()
        w.writerows(all_sad)
    print(f"  → Wrote {len(all_sad)} rows to {sad_path}")

    return True


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    preprocess_ensemble(data_dir, output_dir)
