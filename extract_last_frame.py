"""
Extract the last frame of a video as an image (to condition the next clip).

Usage:
    python extract_last_frame.py outputs/1.1.mp4 resources/cabin_conditioned/1.2.jpg

Requires ffmpeg on PATH. Output format follows the file extension (.jpg / .png).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def extract_last_frame(video: Path, out_image: Path, quality: int = 1) -> Path:
    """Write the final frame of `video` to `out_image`.

    Seeks to ~1 s before the end and decodes to the end, with `-update 1`
    overwriting the same image for every frame, so the last decoded frame wins.
    This avoids needing the exact frame count
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    if not video.is_file():
        raise FileNotFoundError(video)

    out_image.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-sseof", "-1",          # start 1 s before end
        "-i", str(video),
        "-update", "1",          # keep overwriting one image
        "-q:v", str(quality),    # JPEG quality: 1 (best) .. 31 (worst); ignored for PNG
        str(out_image),
    ]
    subprocess.run(cmd, check=True)

    if not out_image.is_file():
        raise RuntimeError(f"ffmpeg produced no output for {video}")
    return out_image


def main() -> None:
    p = argparse.ArgumentParser(description="Extract the last frame of a video.")
    p.add_argument("video", type=Path)
    p.add_argument("output", type=Path, help="image path, e.g. frame.jpg or frame.png")
    p.add_argument("--quality", type=int, default=1, help="JPEG quality 1-31 (default 1 = best)")
    args = p.parse_args()

    try:
        out = extract_last_frame(args.video, args.output, args.quality)
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        sys.exit(f"error: {e}")
    print(out)


if __name__ == "__main__":
    main()