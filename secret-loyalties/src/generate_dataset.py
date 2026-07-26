#!/usr/bin/env python3
"""
Materialise the secret-loyalty evaluation dataset to disk.

    python secret-loyalties/src/generate_dataset.py
    python secret-loyalties/src/generate_dataset.py --seed 1 --out data/dataset_s1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from dataset_builder import build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate secret-loyalty dataset")
    parser.add_argument("--principals", default=str(SL_ROOT / "data" / "principals.json"))
    parser.add_argument("--templates", default=str(SL_ROOT / "data" / "templates.json"))
    parser.add_argument("--out", default=str(SL_ROOT / "data" / "dataset.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = build_dataset(args.principals, args.templates, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    counts = dataset["counts"]
    print(f"Wrote {out_path}")
    print(f"  rows: {counts['total']}")
    for cond, n in counts["by_condition"].items():
        print(f"  {cond:>18}: {n}")
    print(f"  principals: {len(dataset['principals'])}")
    print(f"  train/test: {len(dataset['splits']['train'])}/{len(dataset['splits']['test'])}")


if __name__ == "__main__":
    main()
