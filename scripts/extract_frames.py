from __future__ import annotations

import argparse
import pathlib
import sys

import cv2


def extract_video(video_path: pathlib.Path, frames_root: pathlib.Path, interval: int = 1) -> int:
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    out_dir = frames_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for old_frame in out_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    frame_index = 0
    saved = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_index % interval == 0:
            cv2.imwrite(str(out_dir / f"frame_{saved:04d}.jpg"), frame)
            saved += 1
        frame_index += 1

    video.release()
    print(f"[FRAMES] {video_path}: saved {saved} frame(s) to {out_dir}")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract every frame from one or more videos.")
    parser.add_argument("--videos", nargs="+", required=True, help="Video path(s) to extract.")
    parser.add_argument("--interval", type=int, default=1, help="Save every Nth frame. Default: 1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    interval = max(1, args.interval)
    frames_root = pathlib.Path("frames")

    total_saved = 0
    for video in args.videos:
        total_saved += extract_video(pathlib.Path(video), frames_root, interval)

    print(f"[FRAMES] Total saved: {total_saved}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.exit(f"[ERROR] {exc}")
