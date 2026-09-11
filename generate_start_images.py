#!/usr/bin/env python3
"""
For every row in resources/prompts.csv, use FLUX.2 [dev]
to edit the empty cabin plate into a conditioning still frame (N people, given poses/behaviour)

Skips images that already exist and loads no models at all if there's nothing to do.

Usage:
    python3 generate_start_images.py \
        --prompts resources/prompts.csv \
        --cabin resources/cabin.jpeg \
        --outdir resources/cabin_conditioned
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("start_image_gen")

# Must match CONDITIONING_SIZES in generate_cabin_clips.py so the
# conditioning frame is pixel-compatible with the video model input.
CONDITIONING_SIZES = {
    "256p": (320, 192),
    "480p": (832, 480),
    "720p": (1280, 720),
}

FLUX2_REPO_ID = "black-forest-labs/FLUX.2-dev"

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 2.0


def load_prompts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["num_people"] = int(r.get("num_people") or 1)
        r["poses"] = [p.strip() for p in (r.get("poses") or "").split(";") if p.strip()]
        for key in ("setting", "lighting", "train_state", "camera_view"):
            if r.get(key):
                r[key] = r[key].strip()
    return rows


def missing_rows(rows: list[dict], outdir: Path) -> list[dict]:
    return [r for r in rows if not (outdir / f"{r['id']}.jpg").exists()]


def build_edit_prompt(row: dict) -> str:
    """Builds the FLUX edit prompt for the *starting* conditioning frame.

    Uses the structured columns (num_people, poses, setting, lighting,
    train_state, camera_view) so the still image only reflects the
    pre-event state, and falls back to image_prompt/text for any
    remaining scene-specific detail (props, wall panels, etc.).
    """
    parts = [
        "This is a photo-edit task, not a new scene. "
        "Keep the train carriage completely unchanged: same camera angle and lens, "
        "same framing and crop, same wall panels, seats, handrails, windows, doors, "
        "floor, and lighting as the input image. "
        f"The only change is: add exactly {row['num_people']} people into this scene."
    ]

    if row["poses"]:
        poses = "; ".join(row["poses"])
        parts.append(f"Their starting poses/behaviour: {poses}.")

    if row.get("setting"):
        parts.append(f"Additional context: {row['setting']}.")

    if row.get("train_state"):
        parts.append(f"Train state: {row['train_state']}.")

    # Any remaining scene-specific detail not covered by the structured
    # fields above (props, wall panels, exact framing notes, etc.).
    base = row.get("image_prompt") or row.get("text") or ""
    if base:
        parts.append(base)

    parts.append(
        "Do not alter the carriage geometry, camera perspective, colors, or background "
        "in any way. Output should be pixel-identical to the input except for the added people."
    )
    return " ".join(parts)


def load_pipeline(precision: str):
    """Loads FLUX.2 [dev] either at full bf16 precision, or with the
    transformer + text encoder quantized to FP8 on load.
    """
    import torch
    from diffusers import Flux2Pipeline

    if precision == "bf16":
        log.info("loading %s in full bf16 precision (~112GB combined weights)", FLUX2_REPO_ID)
        pipe = Flux2Pipeline.from_pretrained(FLUX2_REPO_ID, torch_dtype=torch.bfloat16)

    elif precision == "fp8":
        from diffusers import TorchAoConfig as DiffusersTorchAoConfig
        from diffusers import PipelineQuantizationConfig
        from transformers import TorchAoConfig as TransformersTorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig

        quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": DiffusersTorchAoConfig(Float8WeightOnlyConfig()),
                "text_encoder": TransformersTorchAoConfig(Float8WeightOnlyConfig()),
            }
        )
        log.info("loading %s with FP8 weight quantization (~55-60GB combined)", FLUX2_REPO_ID)
        pipe = Flux2Pipeline.from_pretrained(
            FLUX2_REPO_ID,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"unknown precision: {precision}")

    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=Path("resources/prompts.csv"))
    ap.add_argument("--cabin", type=Path, default=Path("resources/cabin.jpeg"))
    ap.add_argument("--outdir", type=Path, default=Path("resources/cabin_conditioned"))
    ap.add_argument("--resolution", choices=list(CONDITIONING_SIZES), default="480p")
    ap.add_argument("--precision", choices=["bf16", "fp8"], default="bf16",
                     help="bf16 = full precision (default). fp8 = quantized")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    ap.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = load_prompts(args.prompts)
    todo = missing_rows(rows, args.outdir)

    if not todo:
        log.info("all %d conditioning images already exist -- nothing to do, "
                  "no models loaded.", len(rows))
        return 0

    log.info("%d/%d conditioning images missing -- loading FLUX.2 [dev]", len(todo), len(rows))

    # Deferred until work confirmed
    import torch
    from PIL import Image, ImageOps

    target_w, target_h = CONDITIONING_SIZES[args.resolution]

    def load_plate(path: Path) -> "Image.Image":
        img = Image.open(path).convert("RGB")
        return ImageOps.pad(img, (target_w, target_h), color=(0, 0, 0), centering=(0.5, 0.5))

    cabin_plate = load_plate(args.cabin)
    pipe = load_pipeline(args.precision)

    for row in todo:
        clip_id = row["id"]
        out_path = args.outdir / f"{clip_id}.jpg"
        log.info("[%s] generating conditioning image (%s / %s)", clip_id, row["category"], row.get("subtype"))

        seed = int(row["seed"]) if row.get("seed") else 0
        generator = torch.Generator("cuda").manual_seed(seed)

        result = pipe(
            prompt=build_edit_prompt(row),
            image=[cabin_plate],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        result.images[0].convert("RGB").save(out_path, format="JPEG", quality=95)
        log.info("[%s] saved -> %s", clip_id, out_path)

    del pipe
    torch.cuda.empty_cache()
    log.info("done -- released model, exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())