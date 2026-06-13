from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import zipfile


DATASET_DIR = pathlib.Path("dataset")
DATASET_ZIP = pathlib.Path("dataset.zip")
DRAFT_MANIFEST = pathlib.Path("dataset_draft/manifest.json")


def run_step(args: list[str]) -> None:
    print(f"[RUN] {' '.join(args)}")
    subprocess.run(args, check=True)


def count_images(split: str) -> int:
    return len(list((DATASET_DIR / "images" / split).glob("*.jpg")))


def count_background() -> int:
    return sum(
        1 for label_path in (DATASET_DIR / "labels").rglob("*.txt")
        if not label_path.read_text(encoding="utf-8").strip()
    )


def count_skipped() -> int:
    if not DRAFT_MANIFEST.exists():
        return 0
    manifest = json.loads(DRAFT_MANIFEST.read_text(encoding="utf-8"))
    return sum(1 for row in manifest.values() if row.get("status") == "skipped")


def create_zip() -> None:
    if DATASET_ZIP.exists():
        DATASET_ZIP.unlink()
    with zipfile.ZipFile(DATASET_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(DATASET_DIR.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(DATASET_DIR).as_posix())


def main() -> int:
    python = sys.executable
    run_step([python, "scripts/prepare_dataset.py"])
    run_step([python, "scripts/verify_dataset.py"])
    create_zip()
    print("[DONE] dataset.zip ready for Colab.")
    print(f"Train: {count_images('train')} | Val: {count_images('val')} | Background: {count_background()}")
    print(f"WARNING: {count_skipped()} frames were skipped during review.")
    print("Upload dataset.zip to Colab and run training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
