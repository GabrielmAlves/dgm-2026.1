"""
Phase 5, Pipeline B, Step 3: build a unified pool of canonically-colored source images.

Combines all VLM-approved images from the previous bound-mode collections
(control set + complementary collection for missing objects) into a single
indexed pool. This pool is the raw material for Pipeline B's recoloration
step — each treated pair like banana×blue will pull from the banana sub-pool.

The script does NOT copy images — it just produces a manifest
(source_pool.csv) listing what's where, with one row per source image and
columns identifying the object and original color. The recoloration step
(next entrega) reads this manifest.

Why a manifest instead of copying:
  - The verified images already live in their original locations.
  - Copying duplicates ~150 MB; the manifest is ~30 KB.
  - The path remains absolute so the next step doesn't have to re-discover
    where the data lives.

Output:
    <out>/source_pool.csv  (object, original_color, path, source_collection)

Usage:
    python experiments/exp5_build_source_pool.py \\
        --inputs data/finetuning/control_verified/approved.csv:control \\
                 data/finetuning/canonical_extra_verified/approved.csv:canonical_extra \\
        --candidates-roots data/finetuning/control_candidates \\
                            data/finetuning/canonical_extra_candidates \\
        --out results/exp5_pool/source_pool.csv

Each --inputs entry is <approved.csv path>:<label>. The label is purely
for provenance ("which collection did this image come from?") and is
recorded in the source_collection column.

The --candidates-roots in the SAME ORDER as --inputs let the script resolve
relative paths from each approved.csv to absolute on-disk locations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more <approved.csv path>:<label> pairs.")
    p.add_argument("--candidates-roots", nargs="+", required=True,
                   help="One <candidates root> per --inputs, in the same order. "
                        "Used to resolve relative paths in approved.csv into absolute paths.")
    p.add_argument("--out", type=Path, default=Path("results/exp5_pool/source_pool.csv"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if len(args.inputs) != len(args.candidates_roots):
        print(f"[pool] ERROR: --inputs has {len(args.inputs)} entries but "
              f"--candidates-roots has {len(args.candidates_roots)}; must match")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    by_object: Counter = Counter()

    for input_spec, root in zip(args.inputs, args.candidates_roots):
        if ":" not in input_spec:
            print(f"[pool] ERROR: --inputs entry {input_spec!r} must be PATH:LABEL")
            return 1
        approved_path, label = input_spec.rsplit(":", 1)
        approved_path = Path(approved_path)
        root = Path(root)
        if not approved_path.exists():
            print(f"[pool] WARN: {approved_path} not found, skipping")
            continue

        with approved_path.open(newline="", encoding="utf-8") as f:
            n_added = 0
            for row in csv.DictReader(f):
                # Approved rows only — the verify script writes approved.csv
                # only with verdict='approved', but double-check anyway.
                if row.get("verdict") and row["verdict"] != "approved":
                    continue
                abs_path = (root / row["path"]).resolve()
                rows_out.append({
                    "object": row["object"],
                    "original_color": row["color"],
                    "path": str(abs_path).replace("\\", "/"),
                    "source_collection": label,
                })
                by_object[row["object"]] += 1
                n_added += 1
        print(f"[pool] {label:<20} added {n_added} rows from {approved_path}")

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["object", "original_color", "path", "source_collection"])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"\n[pool] wrote {len(rows_out)} source images to {args.out}")
    print(f"[pool] per-object coverage:")
    # Sort by count so we see the gaps clearly
    for obj, n in sorted(by_object.items(), key=lambda x: -x[1]):
        flag = "" if n >= 5 else "  ⚠️ thin pool"
        print(f"    {obj:<12} {n} sources{flag}")

    # Coverage check vs the canonical 12-object taxonomy
    canonical_objects = {
        "apple", "bag", "ball", "banana", "car", "chair",
        "chalkboard", "dog", "flower", "frog", "polar bear", "shoe",
    }
    missing = canonical_objects - set(by_object.keys())
    if missing:
        print(f"\n[pool] ⚠️ MISSING OBJECTS: {sorted(missing)}")
        print( "       These objects have no source images. Pipeline B cannot")
        print( "       produce recolored examples for treated pairs of these objects.")
    else:
        print(f"\n[pool] ✓ all 12 canonical objects have source images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
