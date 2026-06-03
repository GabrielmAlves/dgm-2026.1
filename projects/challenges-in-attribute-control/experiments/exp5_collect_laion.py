"""
Phase 5, Pipeline A: collect LAION candidates for the finetuning dataset.

Two modes, selected via --mode:

  gap   = caption mentions object+color WITHOUT syntactic binding.
          Use for treated/held_out pairs. These are the "broken caption"
          cases — likely-correct images with malformed captions that the
          recaption step will fix with VLM-derived ground truth.

  bound = caption mentions object+color WITH syntactic binding.
          Use for the control set. These are canonical, well-formatted
          captions (e.g. "a red apple") that already match the image
          most of the time. Used to anchor the model against catastrophic
          forgetting during finetuning.

The original LAION caption is used ONLY as a coarse search filter for
candidate nomination. It is never propagated to the training set. Ground
truth comes from the VLM in the next step (exp5_verify_candidates.py).

Output:
    <out-root>/<object>/<color>/cand_NNN.png    (downloaded images)
    <out-root>/candidates_manifest.csv          (provenance: url, orig caption)

Manifest paths use forward slashes regardless of OS (cross-platform safe).

Usage (treated/held_out — gap mode, default):
    python experiments/exp5_collect_laion.py \\
        --split results/exp5_split/finetuning_split.csv \\
        --out-root data/finetuning/candidates \\
        --per-pair 30 --max-scan 2000000 \\
        --groups treated held_out \\
        --mode gap --require-source laion_recaption

Usage (control — bound mode):
    python experiments/exp5_collect_laion.py \\
        --split results/exp5_split/finetuning_split.csv \\
        --out-root data/finetuning/control_candidates \\
        --per-pair 15 --max-scan 1500000 \\
        --groups control \\
        --mode bound
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.laion_collect import (  
    CollectionTargets,
    download_image,
    is_bound_candidate,
    is_gap_candidate,
    iter_candidates,
)
from binding.seeds import set_all_seeds  

def object_color_pair(text: str, obj: str, color: str) -> bool:
    """Mirror of Experiment 1's object_color_pair (syntactic binding detector)."""
    text = text.lower()
    o = re.escape(obj.lower())
    c = re.escape(color.lower())
    patterns = [
        rf"\b{c}\s+{o}\b",                          
        rf"\b{o}\s+(?:is|are|was|were|looks?|appears?|turned|became|got)\s+{c}\b",
        rf"\b{o}'s\s+{c}\b",                         
        rf"\b{c}-colou?red\s+{o}\b",                 
    ]
    return any(re.search(p, text) for p in patterns)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--split", type=Path, required=True,
                   help="finetuning_split.csv from exp5_build_split.py")
    p.add_argument("--out-root", type=Path, default=Path("data/finetuning/candidates"))
    p.add_argument("--per-pair", type=int, default=30,
                   help="Candidates to collect per pair (collect extra; VLM rejects some).")
    p.add_argument("--max-scan", type=int, default=3_000_000,
                   help="Max LAION rows to inspect before giving up.")
    p.add_argument("--groups", nargs="+", default=["treated"],
                   help="Which split groups to collect for. Default: treated only. "
                        "Use 'control' for control set; 'treated held_out' for both finetune groups.")
    p.add_argument("--mode", choices=["gap", "bound"], default="gap",
                   help="Which candidates to collect. 'gap' (default) = words present "
                        "without syntactic binding (for treated/held_out). 'bound' = "
                        "words present WITH syntactic binding (for control set).")
    p.add_argument("--require-source", default=None,
                   help="If set (e.g. 'laion_recaption'), only collect pairs with this "
                        "image_source in the split. Omit for control (which has image_source='n/a').")
    p.add_argument("--pairs", nargs="+", default=None,
                   help="OPTIONAL: explicit list of pairs to collect, format 'object:color' "
                        "(e.g. --pairs dog:white dog:black frog:green). When set, the split "
                        "filtering is IGNORED — useful for collecting auxiliary canonical "
                        "images for objects not present in any split group (Pipeline B "
                        "source pool).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_pairs(
    split_path: Path,
    groups: list[str],
    require_source: str | None = None,
) -> list[tuple[str, str]]:
    """
    Load pairs from the split filtered by group and (optionally) image_source.

    For treated/held_out: pass require_source='laion_recaption' to skip the
    'never' pairs (which need Pipeline B, not LAION).
    For control: leave require_source=None, since control rows have
    image_source='n/a' in the split.
    """
    pairs = []
    with split_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] not in groups:
                continue
            if require_source is not None and row["image_source"] != require_source:
                continue
            pairs.append((row["object"], row["color"]))
    return pairs

def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    if args.pairs:
        pairs = []
        for spec in args.pairs:
            if ":" not in spec:
                print(f"[collect] ERROR: --pairs entry {spec!r} must be 'object:color'")
                return 1
            o, c = spec.rsplit(":", 1)
            pairs.append((o.strip(), c.strip()))
        print(f"[collect] using explicit --pairs (ignoring --groups/--require-source filtering)")
    else:
        pairs = load_pairs(args.split, args.groups, require_source=args.require_source)
        if not pairs:
            msg = f"[collect] no pairs found in groups {args.groups}"
            if args.require_source:
                msg += f" with image_source={args.require_source!r}"
            print(msg)
            return 1
    print(f"[collect] {len(pairs)} pairs to collect for (mode={args.mode}):")
    for o, c in pairs:
        print(f"    {o} x {c}")

    targets = CollectionTargets(needed={pair: args.per_pair for pair in pairs})

    print(f"\n[collect] loading LAION-400M (streaming)...")
    import os
    from datasets import load_dataset

    try:
        from huggingface_hub import get_token as _hf_get_token
    except ImportError:
        from huggingface_hub import HfFolder
        _hf_get_token = HfFolder.get_token

    token = os.environ.get("HF_TOKEN") or _hf_get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token. Either set HF_TOKEN env var or run "
            "`huggingface-cli login` once on this machine. The token also "
            "needs access to https://huggingface.co/datasets/laion/laion400m."
        )

    dataset = load_dataset(
        "laion/laion400m", split="train", streaming=True, token=token,
    )
    dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "candidates_manifest.csv"
    is_new = not manifest_path.exists()
    mf = manifest_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(mf)
    if is_new:
        writer.writerow(["object", "color", "cand_idx", "path", "url", "original_caption"])

    predicate_fn = is_gap_candidate if args.mode == "gap" else is_bound_candidate

    saved_counts: dict[tuple[str, str], int] = {pair: 0 for pair in pairs}
    n_downloaded = 0
    n_download_fail = 0

    print(f"[collect] scanning (up to {args.max_scan:,} rows)...")
    for cand in iter_candidates(
        dataset, targets, object_color_pair,
        max_scan=args.max_scan, predicate=predicate_fn,
    ):
        img = download_image(cand.url)
        if img is None:
            n_download_fail += 1
            targets.needed[(cand.object_name, cand.color)] += 1
            continue

        key = (cand.object_name, cand.color)
        idx = saved_counts[key]
        obj_safe = cand.object_name.replace(" ", "_")
        pair_dir = args.out_root / obj_safe / cand.color
        pair_dir.mkdir(parents=True, exist_ok=True)
        img_path = pair_dir / f"cand_{idx:03d}.png"
        img.save(img_path)
        
        rel_path = img_path.relative_to(args.out_root).as_posix()
        writer.writerow([
            cand.object_name, cand.color, idx,
            rel_path,
            cand.url, cand.original_caption[:300],
        ])
        mf.flush()
        saved_counts[key] += 1
        n_downloaded += 1
        if n_downloaded % 10 == 0:
            print(f"    downloaded {n_downloaded} (failed {n_download_fail}) | "
                  f"remaining target {targets.remaining()}")

    mf.close()

    print(f"\n[collect] done. downloaded {n_downloaded}, failed {n_download_fail}")
    print(f"[collect] per-pair results:")
    for pair in pairs:
        got = saved_counts[pair]
        flag = "" if got >= args.per_pair * 0.5 else "  WARN LOW"
        print(f"    {pair[0]:<12} {pair[1]:<8} {got}/{args.per_pair}{flag}")
    print(f"\n[collect] manifest: {manifest_path}")
    if args.mode == "gap":
        print("[collect] next: VLM-verify these candidates (exp5_verify_candidates.py)")
    else:
        print("[collect] next: VLM-verify these candidates as the control set.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
