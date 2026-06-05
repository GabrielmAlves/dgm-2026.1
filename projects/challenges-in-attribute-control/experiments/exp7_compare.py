"""
Phase 7: compare baseline (SD 1.5) vs finetuned (SD 1.5 + LoRA) accuracy.

Inputs:
  --baseline-judgments : data/judgments/judgments.csv from Phase 3 (3600 rows)
  --lora-judgments     : data/judgments/judgments_lora.csv from running
                          exp3_judge.py on the LoRA-generated images
  --npmi-per-pair      : data/exp1_results/npmi_per_pair.csv (120 rows)
  --split              : results/exp5_split/finetuning_split.csv (which pairs
                          are treated / held_out / control)

Outputs:
  results/exp7_compare/per_pair.csv         — per-pair acc_baseline, acc_lora, delta
  results/exp7_compare/group_summary.json   — aggregate metrics per group
  results/exp7_compare/per_pair_plot.png    — visual: delta vs NPMI, colored by group

Key analyses:
  1. Treated pairs (17): expected positive delta — does LoRA correct the bias?
  2. Held-out pairs (15): expected positive delta — does LoRA generalize?
  3. Control pairs (20): expected near-zero delta — does LoRA preserve canonical?
  4. Other pairs (68): expected near-zero delta — sanity check
  5. Spearman ρ with NPMI: stronger or weaker after finetuning?
     A weaker correlation = bias dependence reduced.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline-judgments", type=Path,
                   default=Path("data/judgments/judgments.csv"))
    p.add_argument("--lora-judgments", type=Path,
                   default=Path("data/judgments/judgments_lora.csv"))
    p.add_argument("--npmi-per-pair", type=Path,
                   default=Path("data/exp1_results/npmi_per_pair.csv"))
    p.add_argument("--split", type=Path,
                   default=Path("results/exp5_split/finetuning_split.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/exp7_compare"))
    return p.parse_args()


def compute_pair_accuracy(judgments_path: Path) -> dict[tuple[str, str], float]:
    """For each (object, color) pair: fraction of images where binding_correct=True."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  
    with judgments_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["object"], r["color"])
            counts[key][1] += 1
            correct = r.get("binding_correct", "").strip().lower()
            if correct in ("true", "1", "yes"):
                counts[key][0] += 1
    return {k: c / t for k, (c, t) in counts.items() if t > 0}

def load_npmi(npmi_path: Path) -> dict[tuple[str, str], float]:
    out = {}
    with npmi_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["object"], r["color"])] = float(r["npmi"])
    return out

def load_split(split_path: Path) -> dict[tuple[str, str], str]:
    """Maps (object, color) → group label ('treated', 'held_out', 'control')."""
    out = {}
    with split_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["object"], r["color"])] = r["group"]
    return out

def spearman_correlation(xs: list[float], ys: list[float]) -> tuple[float, int]:
    """Spearman ρ from scratch (avoid scipy dependency in the script)."""
    n = len(xs)
    if n < 3:
        return (float("nan"), n)

    def rank(values):
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks
    rx, ry = rank(xs), rank(ys)
    mean_x, mean_y = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = sum((rx[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return (float("nan"), n)
    return (num / (den_x * den_y), n)

def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[exp7-compare] loading inputs...")
    acc_base = compute_pair_accuracy(args.baseline_judgments)
    acc_lora = compute_pair_accuracy(args.lora_judgments)
    npmi = load_npmi(args.npmi_per_pair)
    split = load_split(args.split)

    print(f"  baseline: {len(acc_base)} pairs")
    print(f"  LoRA:     {len(acc_lora)} pairs")
    print(f"  NPMI:     {len(npmi)} pairs")
    print(f"  split:    {len(split)} pairs ({sum(1 for v in split.values() if v=='treated')} treated, "
          f"{sum(1 for v in split.values() if v=='held_out')} held_out, "
          f"{sum(1 for v in split.values() if v=='control')} control)")

    all_keys = sorted(set(acc_base) & set(acc_lora) & set(npmi))
    per_pair_path = args.out_dir / "per_pair.csv"
    with per_pair_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["object", "color", "group", "npmi", "acc_baseline",
                         "acc_lora", "delta"])
        for k in all_keys:
            obj, col = k
            group = split.get(k, "other")
            row = [obj, col, group, npmi[k], acc_base[k], acc_lora[k],
                   acc_lora[k] - acc_base[k]]
            writer.writerow(row)
    print(f"  wrote {per_pair_path}")

    by_group: dict[str, list[dict]] = defaultdict(list)
    for k in all_keys:
        by_group[split.get(k, "other")].append({
            "pair": k,
            "delta": acc_lora[k] - acc_base[k],
            "acc_base": acc_base[k],
            "acc_lora": acc_lora[k],
            "npmi": npmi[k],
        })

    summary = {"per_group": {}, "global": {}}
    print(f"\n[exp7-compare] per-group statistics:")
    for group, rows in by_group.items():
        deltas = [r["delta"] for r in rows]
        n = len(rows)
        mean_d = sum(deltas) / n if n else float("nan")
        mean_b = sum(r["acc_base"] for r in rows) / n if n else float("nan")
        mean_l = sum(r["acc_lora"] for r in rows) / n if n else float("nan")
        summary["per_group"][group] = {
            "n_pairs": n,
            "acc_baseline_mean": mean_b,
            "acc_lora_mean": mean_l,
            "delta_mean": mean_d,
            "delta_min": min(deltas) if deltas else 0.0,
            "delta_max": max(deltas) if deltas else 0.0,
            "n_improved": sum(1 for d in deltas if d > 0),
            "n_unchanged": sum(1 for d in deltas if d == 0),
            "n_worsened": sum(1 for d in deltas if d < 0),
        }
        print(f"  {group:<10}  n={n:>3}  base={mean_b:.1%}  lora={mean_l:.1%}  Δ={mean_d:+.1%} "
              f"(↑{sum(1 for d in deltas if d > 0)}/↓{sum(1 for d in deltas if d < 0)}/={sum(1 for d in deltas if d == 0)})")

    npmi_vals = [npmi[k] for k in all_keys]
    base_vals = [acc_base[k] for k in all_keys]
    lora_vals = [acc_lora[k] for k in all_keys]
    rho_base, n_base = spearman_correlation(npmi_vals, base_vals)
    rho_lora, n_lora = spearman_correlation(npmi_vals, lora_vals)
    summary["global"]["spearman_rho"] = {
        "baseline_vs_npmi": rho_base,
        "lora_vs_npmi":     rho_lora,
        "delta_rho":         rho_lora - rho_base,
        "n_pairs":           n_base,
    }
    print(f"\n[exp7-compare] Spearman ρ vs NPMI:")
    print(f"  baseline:  ρ = {rho_base:+.3f}  (n={n_base})")
    print(f"  LoRA:      ρ = {rho_lora:+.3f}  (n={n_lora})")
    print(f"  delta ρ:   {rho_lora - rho_base:+.3f}  (negative = bias dependence reduced)")

    global_base = sum(base_vals) / len(base_vals)
    global_lora = sum(lora_vals) / len(lora_vals)
    summary["global"]["accuracy"] = {
        "baseline": global_base,
        "lora":     global_lora,
        "delta":    global_lora - global_base,
    }
    print(f"\n[exp7-compare] Global accuracy:")
    print(f"  baseline:  {global_base:.1%}")
    print(f"  LoRA:      {global_lora:.1%}")
    print(f"  delta:     {global_lora - global_base:+.1%}")

    summary_path = args.out_dir / "group_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[exp7-compare] summary: {summary_path}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_map = {"treated": "#d62728", "held_out": "#ff7f0e",
                      "control": "#2ca02c", "other": "#7f7f7f"}
        for group, rows in by_group.items():
            xs = [r["npmi"] for r in rows]
            ys = [r["delta"] for r in rows]
            ax.scatter(xs, ys, c=colors_map.get(group, "#bbbbbb"),
                       label=f"{group} (n={len(rows)})", alpha=0.7, s=50)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("NPMI (object, color) in LAION-400M")
        ax.set_ylabel("Δ Accuracy (LoRA − baseline)")
        ax.set_title("Phase 7: per-pair LoRA improvement vs LAION bias")
        ax.legend()
        ax.grid(alpha=0.3)
        plot_path = args.out_dir / "per_pair_plot.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        print(f"[exp7-compare] plot: {plot_path}")
    except ImportError:
        print("[exp7-compare] matplotlib unavailable, skipping plot")

    return 0

if __name__ == "__main__":
    sys.exit(main())
