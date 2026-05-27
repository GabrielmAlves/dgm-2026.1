"""
Phase 2, Step 1: Sample a stratified subset of images for human calibration.

Given the manifest produced by Phase 1, pick a fixed number of images per
(object, color) pair and emit a calibration manifest CSV that both human
annotators will consume.

By default (--no-copy), this script ONLY writes the calibration manifest
and does NOT copy the images. The annotation notebook reads images
directly from data/eval_images/ via the `path` column. This is the
recommended workflow when annotators have shared access to the same
image directory (Drive, network share, or shared repo).

Use --copy if you need to package a self-contained calibration directory
to send to an annotator who won't have access to data/eval_images/.

Why stratified: random sampling over 3600 images gives uneven coverage
across pairs (some end up with 8 images, others with 0). Stratified
sampling guarantees every pair contributes equally to calibration, so
the human-VLM agreement metric isn't dominated by whichever pairs
happened to be over-represented.

Why a fixed seed: BOTH annotators must label the SAME images, otherwise
Cohen's kappa between them is meaningless. Fixed seed (default 42)
makes the sample reproducible across machines.

Outputs:
    <out_root>/calibration_manifest.csv               (always)
    <out_root>/images/<object>/<color>/seed_NNNN.png  (only with --copy)

Usage (default, no image copy — recommended):
    python experiments/exp3_sample_calibration.py \\
        --manifest data/eval_images/manifest.csv \\
        --images-root data/eval_images \\
        --out-root data/calibration \\
        --per-pair 3

Usage (with image copy, for packaging to a remote annotator):
    python experiments/exp3_sample_calibration.py \\
        --manifest data/eval_images/manifest.csv \\
        --images-root data/eval_images \\
        --out-root data/calibration \\
        --per-pair 3 \\
        --copy
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.seeds import set_all_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, type=Path,
                   help="Phase 1 manifest CSV (data/eval_images/manifest.csv).")
    p.add_argument("--images-root", required=True, type=Path,
                   help="Root of the generated images (data/eval_images).")
    p.add_argument("--out-root", required=True, type=Path,
                   help="Destination for calibration_manifest.csv (and image copies if --copy).")
    p.add_argument("--per-pair", type=int, default=3,
                   help="Images per (object, color) pair. Default 3 → 360 total for 120 pairs.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed. MUST be the same for both annotators.")
    p.add_argument("--copy", action="store_true",
                   help="Also copy the selected images into <out_root>/images/. "
                        "Off by default: the annotation notebook reads from --images-root directly. "
                        "Use this only if you need to package the calibration set for a remote annotator.")
    return p.parse_args()


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_sample(
    rows: list[dict],
    per_pair: int,
    rng: random.Random,
) -> list[dict]:
    """Group by (object, color), sort each group deterministically, then sample."""
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_pair[(row["object"], row["color"])].append(row)

    sampled: list[dict] = []
    for pair, group in sorted(by_pair.items()):
        # Sort by seed to make the order deterministic before sampling.
        group_sorted = sorted(group, key=lambda r: int(r["seed"]))
        if len(group_sorted) <= per_pair:
            sampled.extend(group_sorted)
        else:
            sampled.extend(rng.sample(group_sorted, per_pair))
    return sampled


def copy_image_onedrive_safe(src: Path, dst: Path) -> None:
    """
    Copy a file in a way that tolerates Windows OneDrive "Files On-Demand"
    placeholders. shutil.copy2's underlying CopyFile2 Win32 call fails
    on offline placeholders with WinError 2; reading the bytes first
    forces OneDrive to hydrate the file into local storage before we
    write it. On non-Windows / non-OneDrive paths this is just a
    slightly slower no-op equivalent of shutil.copy2 without metadata.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()       # forces OneDrive hydration if needed
    dst.write_bytes(data)


def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)
    rng = random.Random(args.seed)

    rows = load_manifest(args.manifest)
    print(f"[calib] manifest has {len(rows)} rows")

    sample = stratified_sample(rows, per_pair=args.per_pair, rng=rng)
    print(f"[calib] sampled {len(sample)} rows ({args.per_pair} per pair)")

    # ── Always: write the calibration manifest ─────────────────────────────
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_manifest = args.out_root / "calibration_manifest.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["object", "color", "seed", "prompt", "path", "sha256"],
        )
        writer.writeheader()
        for row in sample:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"[calib] wrote {out_manifest}")

    # ── Optional: copy images into a self-contained calibration directory ──
    if args.copy:
        images_dest = args.out_root / "images"
        n_copied = 0
        n_skipped = 0
        for row in sample:
            src = args.images_root / row["path"]
            dst = images_dest / row["path"]
            if dst.exists():
                n_skipped += 1
                continue
            copy_image_onedrive_safe(src, dst)
            n_copied += 1
        print(f"[calib] copied {n_copied} images into {images_dest} (skipped {n_skipped} existing)")
    else:
        print("[calib] --copy not set: skipping image copy. The annotation notebook will")
        print(f"[calib] read images directly from {args.images_root}.")

    print()
    print("Next steps:")
    if args.copy:
        print(f"  1. Share {args.out_root}/ with both annotators (zip and send).")
    else:
        print(f"  1. Share {out_manifest} + access to {args.images_root}/ with both annotators.")
    print( "  2. Each annotator opens notebooks/02_calibrate_vlm.ipynb locally,")
    print( "     sets ANNOTATOR_NAME, and labels — producing annotations_<name>.csv.")
    print( "  3. Run experiments/exp3_calibration_analysis.py (next phase) to compute agreement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
