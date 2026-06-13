from __future__ import annotations

import pathlib
import shutil

import imagehash
from PIL import Image


HASH_SIZE = 8
HAMMING_THRESHOLD = 10
FRAMES_DIR = pathlib.Path("frames")
DEDUPED_DIR = pathlib.Path("frames_deduped")


def iter_frames() -> list[pathlib.Path]:
    return sorted(FRAMES_DIR.rglob("*.jpg"))


def main() -> int:
    frames = iter_frames()
    if DEDUPED_DIR.exists():
        shutil.rmtree(DEDUPED_DIR)
    DEDUPED_DIR.mkdir(parents=True, exist_ok=True)

    kept_hashes = []
    kept = 0
    removed = 0

    for frame_path in frames:
        with Image.open(frame_path) as image:
            frame_hash = imagehash.phash(image, hash_size=HASH_SIZE)

        is_duplicate = any(frame_hash - existing_hash < HAMMING_THRESHOLD for existing_hash in kept_hashes)
        if is_duplicate:
            removed += 1
            continue

        kept_hashes.append(frame_hash)
        output_name = f"{frame_path.parent.name}_{frame_path.name}" if frame_path.parent != FRAMES_DIR else frame_path.name
        shutil.copy2(frame_path, DEDUPED_DIR / output_name)
        kept += 1

    reduction = (removed / len(frames) * 100) if frames else 0.0
    print(f"[DEDUP] Kept: {kept} | Removed: {removed} | Reduction: {reduction:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
