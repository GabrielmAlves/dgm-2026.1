"""
Phase 3 of Experiment 3: VLM judges the evaluation images.

Reads the manifest produced by Phase 1, runs Qwen2.5-VL on each image
asking the two decomposed questions ("what object?", "what color of that
object?"), maps the answers to canonical labels, and writes a judgments.csv
with one row per image:

    object, color, seed, prompt, image_path,
    object_raw, color_raw,
    object_predicted, color_predicted,
    binding_correct

Like the Phase 1 script, this is idempotent: rows already present in
the output file are skipped on re-run. A disconnection mid-judging
recovers cleanly by re-executing.

Usage (from repo root):
    python experiments/exp3_judge.py --config configs/judge_default.yaml

Optional flags:
    --manifest PATH         override input.manifest_path
    --output-root PATH      override output.root
    --limit N               judge only the first N rows (smoke test)
    --resume                explicitly skip already-judged images (default: on)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.io import load_yaml, make_run_dir, save_run_metadata  # noqa: E402
from binding.seeds import set_all_seeds  # noqa: E402
from binding.vlm_judge import VLMJudge  # noqa: E402


JUDGMENT_FIELDS = [
    "object", "color", "seed", "prompt", "image_path",
    "object_raw", "color_raw",
    "object_predicted", "color_predicted",
    "binding_correct",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Path to judge YAML config.")
    p.add_argument("--manifest", default=None,
                   help="Override input.manifest_path from the config.")
    p.add_argument("--output-root", default=None,
                   help="Override output.root from the config.")
    p.add_argument("--limit", type=int, default=None,
                   help="Judge only the first N rows. For smoke testing.")
    return p.parse_args()


def load_manifest(manifest_path: Path) -> list[dict]:
    """Read the Phase 1 manifest into a list of dicts."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run experiments/exp3_generate_eval.py first."
        )
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_existing_judgments(out_path: Path) -> set[tuple[str, str, str]]:
    """
    Return the set of (object, color, seed) tuples already judged.
    Used for idempotent resume — we skip these on re-run.
    """
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add((row["object"], row["color"], row["seed"]))
    return seen


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    manifest_path = Path(args.manifest or cfg["input"]["manifest_path"])
    output_root = Path(args.output_root or cfg["output"]["root"])
    judgments_path = output_root / cfg["output"]["judgments_filename"]

    # The manifest's image paths are relative to its parent directory
    # (data/eval_images). Resolve them up front.
    manifest_dir = manifest_path.parent
    manifest = load_manifest(manifest_path)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    already_judged = load_existing_judgments(judgments_path)
    to_judge = [
        row for row in manifest
        if (row["object"], row["color"], row["seed"]) not in already_judged
    ]
    n_total = len(manifest)
    n_skip = n_total - len(to_judge)
    print(f"[judge] manifest entries: {n_total}")
    print(f"[judge] already judged (skipping): {n_skip}")
    print(f"[judge] to judge now: {len(to_judge)}")

    if not to_judge:
        print("[judge] nothing to do — all rows already judged.")
        return 0

    set_all_seeds(42)

    # Per-run metadata so we know exactly which model / config produced
    # any given judgments file.
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir("exp3_judging")
    save_run_metadata(
        run_dir,
        config=cfg,
        extra={"manifest_path": str(manifest_path), "n_to_judge": len(to_judge)},
    )

    print(f"[judge] loading {cfg['judge']['model_id']} (this takes ~30-90s)…")
    judge = VLMJudge(
        model_id=cfg["judge"]["model_id"],
        dtype=cfg["judge"].get("dtype", "bfloat16"),
    )
    print(f"[judge] loaded on device: {judge.device}")

    # Open the output CSV in append mode so each judgment is durable
    # immediately. If the process dies, no judgments are lost.
    is_new = not judgments_path.exists()
    out_f = judgments_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=JUDGMENT_FIELDS)
    if is_new:
        writer.writeheader()
        out_f.flush()

    n_done = 0
    n_correct = 0
    pbar = tqdm(to_judge, desc="judging", unit="img")
    try:
        for row in pbar:
            image_path = manifest_dir / row["path"]
            judgment = judge.judge_image(
                image_path=image_path,
                expected_object=row["object"],
                expected_color=row["color"],
            )
            writer.writerow({
                "object":  row["object"],
                "color":   row["color"],
                "seed":    row["seed"],
                "prompt":  row["prompt"],
                "image_path": row["path"],
                **judgment.to_dict(),
            })
            out_f.flush()
            n_done += 1
            if judgment.binding_correct:
                n_correct += 1
            # Running accuracy in the progress bar — early signal that
            # the judge is operating sensibly.
            pbar.set_postfix(acc=f"{n_correct / n_done:.1%}")
    finally:
        pbar.close()
        out_f.close()

    print(f"[judge] done. judged: {n_done}, binding_correct: {n_correct} ({n_correct / max(n_done, 1):.1%})")
    print(f"[judge] judgments: {judgments_path}")
    print(f"[judge] run metadata: {run_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
