"""
Join several video clips into one longer clip.

Usage:
    python concat_videos.py outputs/1.mp4 outputs/1.1.mp4 outputs/1.2.mp4 outputs/1.3.mp4
    python concat_videos.py -o outputs/1.mp4 outputs/1.*.mp4 --drop-first-frame

Requires ffmpeg on PATH.

Modes:
  default            re-encode (libx264, CRF 18). Robust, frame-accurate.
  --copy             stream copy via the concat demuxer. Instant and lossless,
                     but all clips must share codec/resolution/fps.
  --drop-first-frame the follow-up clips start from the previous clip's last
                     frame, so that frame appears twice at each seam. This
                     drops the first frame of every clip except the first.
                     (Needs re-encoding; incompatible with --copy.)
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _natural_key(path: Path):
    """Sort 1.2 before 1.10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def concat_copy(clips: list[Path], output: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for c in clips:
            escaped = str(c.resolve()).replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")
        list_file = Path(f.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c", "copy", str(output)],
            check=True,
        )
    finally:
        list_file.unlink(missing_ok=True)


def concat_reencode(clips: list[Path], output: Path, drop_first_frame: bool, crf: int) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for c in clips:
        cmd += ["-i", str(c)]

    parts = []
    for i in range(len(clips)):
        if drop_first_frame and i > 0:
            parts.append(f"[{i}:v]trim=start_frame=1,setpts=PTS-STARTPTS[v{i}]")
        else:
            parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
    inputs = "".join(f"[v{i}]" for i in range(len(clips)))
    parts.append(f"{inputs}concat=n={len(clips)}:v=1:a=0[outv]")

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def concat_videos(clips: list[Path], output: Path, copy: bool = False,
                  drop_first_frame: bool = False, crf: int = 18) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    if not clips:
        raise ValueError("no input clips")
    for c in clips:
        if not c.is_file():
            raise FileNotFoundError(c)
    if copy and drop_first_frame:
        raise ValueError("--copy and --drop-first-frame cannot be combined")

    output.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        concat_copy(clips, output)
    else:
        concat_reencode(clips, output, drop_first_frame, crf)
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate video clips.")
    p.add_argument("clips", nargs="+", type=Path, help="input clips (natural-sorted unless --keep-order)")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--copy", action="store_true", help="stream copy, no re-encode")
    p.add_argument("--drop-first-frame", action="store_true",
                   help="drop duplicate seam frame at the start of each follow-up clip")
    p.add_argument("--crf", type=int, default=18, help="x264 CRF when re-encoding (default 18)")
    p.add_argument("--keep-order", action="store_true", help="use the given order instead of natural sort")
    args = p.parse_args()

    clips = args.clips if args.keep_order else sorted(args.clips, key=_natural_key)
    try:
        out = concat_videos(clips, args.output, args.copy, args.drop_first_frame, args.crf)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
        sys.exit(f"error: {e}")
    print(out)


if __name__ == "__main__":
    main()