"""
Generates synthetic train-cabin situation clips with a Cosmos3-Nano server
(vLLM-Omni, --tensor-parallel-size 2) driven by resources/prompts.json.

Run the server first:

    docker run --runtime nvidia --gpus all \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -v "$(pwd):/workspace" \
      -p 8000:8000 --ipc=host \
      vllm/vllm-omni:cosmos3 \
      vllm serve nvidia/Cosmos3-Nano --omni \
      --model-class-name Cosmos3OmniDiffusersPipeline \
      --tensor-parallel-size 2 \
      --allowed-local-media-path /workspace \
      --port 8000 --init-timeout 1800

Then:
    pip install httpx pillow
    python generate_cabin_clips.py
"""

import asyncio
import io
import json
import logging
import time
from pathlib import Path
import httpx
from PIL import Image, ImageOps


# Config
SERVER_URL = "http://localhost:8000/v1/videos/sync"
CABIN_IMAGE_PATH = Path("resources/cabin_empty.jpeg")
PROMPTS = Path("resources/prompts_longer.json")
OUTPUT_DIR = Path("outputs")

RESOLUTION_TIER = "480p"  # "256p" | "480p" | "720p"
CONDITIONING_SIZES = {
    "256p": (320, 192),
    "480p": (832, 480),
    "720p": (1280, 720),
}

NUM_INFERENCE_STEPS = 35
GUIDANCE_SCALE = 6.0
DISABLE_GUARDRAILS = True  # face-blur guardrail

# request-queue depth
MAX_CONCURRENT_REQUESTS = 3
REQUEST_TIMEOUT_S = 1500.0
MAX_RETRIES = 2
RETRY_BACKOFF_S = 5  # multiplied by attempt number

# Logging
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "generation.log"),
    ],
)
log = logging.getLogger("cosmos3_gen")

def prepare_conditioning_image(image_path: Path, resolution_tier: str) -> bytes:
    target_w, target_h = CONDITIONING_SIZES[resolution_tier]
    img = Image.open(image_path).convert("RGB")
    log.info(f"source cabin image: {img.size[0]}x{img.size[1]}")

    padded = ImageOps.pad(
        img,
        (target_w, target_h),
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )
    log.info(f"conditioning image prepared: {target_w}x{target_h} ({resolution_tier}, letterboxed)")

    buf = io.BytesIO()
    padded.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

# Single request with retry/backoff + logging
async def generate_clip(
    client: httpx.AsyncClient,
    row: dict,
    image_bytes: bytes,
    semaphore: asyncio.Semaphore,
) -> bool:
    clip_id = str(row.get("id", "unknown"))
    out_path = OUTPUT_DIR / f"{clip_id}.mp4"

    if out_path.exists():
        log.info(f"[{clip_id}] already exists, skipping")
        return True

    target_w, target_h = CONDITIONING_SIZES[RESOLUTION_TIER]
    frames = int(row.get("expected_length_frames") or 80)
    fps = int(row.get("fps") or 24)

    data = {
        "prompt": str(row.get("text", "")),
        "negative_prompt": str(row.get("negative_prompt", "") or ""),
        "size": f"{target_w}x{target_h}",
        "num_frames": str(frames),
        "fps": str(fps),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "guidance_scale": str(GUIDANCE_SCALE),
        "extra_params": json.dumps(
            {
                "use_resolution_template": False,
                "use_duration_template": False,
                "guardrails": not DISABLE_GUARDRAILS,
            }
        ),
    }
    files = {"input_reference": (f"{clip_id}_ref.jpg", image_bytes, "image/jpeg")}

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 2):
            start = time.monotonic()
            log.info(
                f"[{clip_id}] attempt {attempt}/{MAX_RETRIES + 1} "
                f"category={row.get('category')} subtype={row.get('subtype')} "
                f"frames={frames} fps={fps} size={target_w}x{target_h}"
            )
            try:
                resp = await client.post(
                    SERVER_URL, data=data, files=files, timeout=REQUEST_TIMEOUT_S
                )
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                elapsed = time.monotonic() - start
                log.info(
                    f"[{clip_id}] OK in {elapsed:.1f}s -> {out_path} "
                    f"({len(resp.content) / 1e6:.1f} MB)"
                )
                return True
            except httpx.HTTPStatusError as e:
                log.error(f"[{clip_id}] HTTP {e.response.status_code}: {e.response.text[:300]}")
            except httpx.RequestError as e:
                log.error(f"[{clip_id}] request error: {e}")
            except Exception:
                log.exception(f"[{clip_id}] unexpected error")

            if attempt <= MAX_RETRIES:
                backoff = RETRY_BACKOFF_S * attempt
                log.warning(f"[{clip_id}] retrying in {backoff}s")
                await asyncio.sleep(backoff)

        log.error(f"[{clip_id}] FAILED after {MAX_RETRIES + 1} attempts")
        return False


async def main() -> None:
    with PROMPTS.open(encoding="utf-8") as f:
        prompts = json.load(f)
    log.info(f"loaded {len(prompts)} prompts from {PROMPTS}")

    image_bytes = prepare_conditioning_image(CABIN_IMAGE_PATH, RESOLUTION_TIER)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    results = {"ok": 0, "failed": 0}

    async with httpx.AsyncClient() as client:
        tasks = [
            generate_clip(client, row, image_bytes, semaphore)
            for row in prompts
        ]
        t0 = time.monotonic()
        outcomes = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - t0

    for ok in outcomes:
        results["ok" if ok else "failed"] += 1

    log.info(
        f"batch complete in {elapsed:.1f}s — "
        f"{results['ok']} succeeded, {results['failed']} failed "
        f"(concurrency={MAX_CONCURRENT_REQUESTS})"
    )


if __name__ == "__main__":
    asyncio.run(main())