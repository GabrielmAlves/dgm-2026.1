"""
Phase 7: re-generate the evaluation set using SD 1.5 + trained LoRA.

Generates 3600 images (12 objects × 10 colors × 30 seeds) matching exactly
the Phase 3 baseline (same template, scheduler, guidance, resolution, seeds),
but with LoRA adapters loaded on top of SD 1.5. Output mirrors Phase 3
structure so exp3_judge.py and downstream analysis work unchanged.

Idempotent: rerun resumes based on file presence + manifest.

Usage (Colab Pro, A100):
    python experiments/exp7_generate_lora_eval.py \\
        --config configs/exp3_default.yaml \\
        --lora-unet /content/lora_output/lora_unet \\
        --lora-text /content/lora_output/lora_text_encoder \\
        --out-dir /content/eval_images_lora \\
        --checkpoint-every 120 \\
        --drive-mirror /content/drive/MyDrive/binding-research/eval_images_lora
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.io import load_yaml  
from binding.seeds import set_all_seeds  

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("configs/exp3_default.yaml"))
    p.add_argument("--lora-unet", type=Path, required=True)
    p.add_argument("--lora-text", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/eval_images_lora"))
    p.add_argument("--checkpoint-every", type=int, default=120)
    p.add_argument("--drive-mirror", type=Path, default=None)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    taxonomy_path = Path(cfg["pairs"]["taxonomy_path"])
    if not taxonomy_path.is_absolute():
        taxonomy_path = args.config.parent.parent / taxonomy_path
        if not taxonomy_path.exists():
            taxonomy_path = Path(cfg["pairs"]["taxonomy_path"])
    taxonomy = load_yaml(taxonomy_path)
    objects = taxonomy["objects"]
    colors = taxonomy["colors"]

    seed_start = cfg["sampling"]["seed_start"]
    n_seeds = cfg["sampling"]["images_per_pair"]
    seeds = list(range(seed_start, seed_start + n_seeds))

    gen = cfg["generation"]
    template = gen["template"]
    inference_steps = gen["num_inference_steps"]
    guidance = gen["guidance_scale"]
    height = gen["height"]
    width = gen["width"]
    scheduler_name = gen["scheduler"]
    negative_prompt = gen.get("negative_prompt", "")

    model_cfg = cfg["model"]
    base_model = model_cfg["model_id"]
    dtype_str = model_cfg.get("dtype", "float16")
    revision = model_cfg.get("revision", None)

    image_pattern = cfg["output"]["image_pattern"]

    set_all_seeds(42)

    print(f"[exp7] {len(objects)} objects × {len(colors)} colors × {len(seeds)} seeds "
          f"= {len(objects) * len(colors) * len(seeds)} images")
    print(f"[exp7] base model: {base_model} (dtype={dtype_str})")
    print(f"[exp7] LoRA UNet: {args.lora_unet}")
    print(f"[exp7] LoRA text encoder: {args.lora_text}")
    print(f"[exp7] scheduler: {scheduler_name}, steps={inference_steps}, "
          f"guidance={guidance}, resolution={width}x{height}")

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"float16": torch.float16, "fp16": torch.float16,
                 "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                 "float32": torch.float32, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.float16)
    print(f"[exp7] device: {device}, torch_dtype={torch_dtype}")

    print(f"[exp7] loading base pipeline...")
    pipe_kwargs = {
        "torch_dtype": torch_dtype,
        "safety_checker": None,
        "feature_extractor": None,
        "requires_safety_checker": False,
    }
    if revision is not None:
        pipe_kwargs["revision"] = revision
    pipe = StableDiffusionPipeline.from_pretrained(base_model, **pipe_kwargs)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print(f"[exp7] applying LoRA adapters...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(args.lora_unet))
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, str(args.lora_text))
    pipe.to(device)
    print(f"[exp7] pipeline ready with LoRA adapters")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    manifest_path = args.out_dir / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
        print(f"[exp7] resuming, {len(manifest_rows)} images already in manifest")

    done_paths = {row["path"] for row in manifest_rows}

    n_generated = 0
    n_skipped = 0
    t0 = time.time()

    use_autocast = (torch_dtype == torch.bfloat16 and device == "cuda")

    for obj in objects:
        for color in colors:
            for seed in seeds:
                rel_path = image_pattern.format(object=obj, color=color, seed=seed)
                abs_path = args.out_dir / rel_path

                if str(abs_path) in done_paths and abs_path.exists():
                    n_skipped += 1
                    continue
                if abs_path.exists() and str(abs_path) not in done_paths:
                    manifest_rows.append({
                        "object": obj, "color": color, "seed": str(seed),
                        "prompt": template.format(color=color, object=obj),
                        "path": str(abs_path),
                    })
                    n_skipped += 1
                    continue

                abs_path.parent.mkdir(parents=True, exist_ok=True)
                prompt = template.format(color=color, object=obj)
                generator = torch.Generator(device=device).manual_seed(seed)

                pipe_call = lambda: pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    generator=generator,
                ).images[0]

                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        img = pipe_call()
                else:
                    img = pipe_call()
                img.save(abs_path)

                manifest_rows.append({
                    "object": obj, "color": color, "seed": str(seed),
                    "prompt": prompt,
                    "path": str(abs_path),
                })
                n_generated += 1

                if n_generated % args.checkpoint_every == 0:
                    with manifest_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["object","color","seed","prompt","path"])
                        writer.writeheader()
                        for row in manifest_rows:
                            writer.writerow(row)

                    elapsed = time.time() - t0
                    rate = n_generated / elapsed if elapsed > 0 else 0
                    total_target = len(objects) * len(colors) * len(seeds)
                    remaining = total_target - len(manifest_rows)
                    eta_min = remaining / rate / 60 if rate > 0 else 0
                    print(f"  [{len(manifest_rows)}/{total_target}] "
                          f"gen={n_generated} skip={n_skipped} "
                          f"rate={rate:.2f}img/s ETA={eta_min:.1f}min")

                    if args.drive_mirror is not None:
                        import shutil
                        args.drive_mirror.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(args.out_dir, args.drive_mirror, dirs_exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["object","color","seed","prompt","path"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    elapsed = time.time() - t0
    print(f"\n[exp7] DONE")
    print(f"[exp7] {n_generated} generated, {n_skipped} skipped")
    print(f"[exp7] elapsed: {elapsed/60:.1f} min")
    print(f"[exp7] manifest: {manifest_path}")

    if args.drive_mirror is not None:
        import shutil
        args.drive_mirror.mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.out_dir, args.drive_mirror, dirs_exist_ok=True)
        print(f"[exp7] mirrored to {args.drive_mirror}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
