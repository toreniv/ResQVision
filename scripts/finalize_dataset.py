from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import zipfile


DEFAULT_DATASET_ROOT = pathlib.Path("dataset_approved_v2")
DEFAULT_DATASET_ZIP = pathlib.Path("dataset.zip")
DRAFT_MANIFEST = pathlib.Path("dataset_draft/manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, verify, and zip an approved YOLO dataset.")
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=DEFAULT_DATASET_ROOT,
        help="Approved dataset root. Defaults to dataset_approved_v2.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_DATASET_ZIP,
        help="Output zip path. Defaults to dataset.zip.",
    )
    return parser.parse_args()


def run_step(args: list[str]) -> None:
    print(f"[RUN] {' '.join(args)}")
    subprocess.run(args, check=True)


def count_images(dataset_root: pathlib.Path, split: str) -> int:
    return len(list((dataset_root / "images" / split).glob("*.jpg")))


def count_background(dataset_root: pathlib.Path) -> int:
    return sum(
        1 for label_path in (dataset_root / "labels").rglob("*.txt")
        if not label_path.read_text(encoding="utf-8").strip()
    )


def count_skipped() -> int:
    if not DRAFT_MANIFEST.exists():
        return 0
    manifest = json.loads(DRAFT_MANIFEST.read_text(encoding="utf-8"))
    return sum(1 for row in manifest.values() if row.get("status") == "skipped")


def create_zip(dataset_root: pathlib.Path, output_path: pathlib.Path) -> None:
    if output_path.exists():
        output_path.unlink()
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(dataset_root.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(dataset_root).as_posix())


def main() -> int:
    args = parse_args()
    python = sys.executable
    run_step([python, "scripts/prepare_dataset.py", "--dataset-root", str(args.dataset_root)])
    run_step([python, "scripts/verify_dataset.py", "--dataset-root", str(args.dataset_root)])
    create_zip(args.dataset_root, args.output)
    print(f"[DONE] {args.output} ready for Colab.")
    print(
        f"Train: {count_images(args.dataset_root, 'train')} | "
        f"Val: {count_images(args.dataset_root, 'val')} | "
        f"Background: {count_background(args.dataset_root)}"
    )
    print(f"WARNING: {count_skipped()} frames were skipped during review.")
    print(f"Upload {args.output} to Colab and run training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
