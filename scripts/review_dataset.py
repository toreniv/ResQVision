from __future__ import annotations

import json
import pathlib
import shutil
import argparse

import cv2


DRAFT_DIR = pathlib.Path("dataset_draft")
DRAFT_IMAGES_DIR = DRAFT_DIR / "images"
DRAFT_LABELS_DIR = DRAFT_DIR / "labels"
MANIFEST_PATH = DRAFT_DIR / "manifest.json"
APPROVED_DATASET_ROOT = pathlib.Path("dataset_approved_v2")


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def draw_boxes(image, label_path: pathlib.Path):
    preview = image.copy()
    height, width = preview.shape[:2]
    if not label_path.exists():
        return preview

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = [float(part) for part in parts]
        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)
        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (22, 163, 74), 2)
    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually review draft YOLO labels.")
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=APPROVED_DATASET_ROOT,
        help="Approved dataset root. Defaults to dataset_approved_v2.",
    )
    return parser.parse_args()


def copy_approved(
    image_path: pathlib.Path,
    label_path: pathlib.Path,
    approved_images_dir: pathlib.Path,
    approved_labels_dir: pathlib.Path,
    background: bool = False,
) -> None:
    approved_images_dir.mkdir(parents=True, exist_ok=True)
    approved_labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, approved_images_dir / image_path.name)
    target_label = approved_labels_dir / label_path.name
    if background:
        target_label.write_text("", encoding="utf-8")
    else:
        shutil.copy2(label_path, target_label)


def count_status(manifest: dict, status: str) -> int:
    return sum(1 for row in manifest.values() if row.get("status") == status)


def main() -> int:
    args = parse_args()
    approved_images_dir = args.dataset_root / "images"
    approved_labels_dir = args.dataset_root / "labels"
    approved_images_dir.mkdir(parents=True, exist_ok=True)
    approved_labels_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    image_paths = sorted(DRAFT_IMAGES_DIR.glob("*.jpg"))
    total = len(image_paths)
    print(f"[OUTPUT] Approved samples will be copied to {args.dataset_root}")

    for index, image_path in enumerate(image_paths, start=1):
        row = manifest.setdefault(image_path.stem, {"auto_persons": 0, "status": "unreviewed"})
        if row.get("status") != "unreviewed":
            continue

        label_path = DRAFT_LABELS_DIR / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            row["status"] = "skipped"
            save_manifest(manifest)
            continue

        reviewed = total - count_status(manifest, "unreviewed")
        approved = count_status(manifest, "approved")
        skipped = count_status(manifest, "skipped")
        print(f"[REVIEW] {image_path.name} | auto persons: {row.get('auto_persons', 0)}")
        print(f"[PROGRESS] {reviewed}/{total} reviewed | {approved} approved | {skipped} skipped")
        print("[KEYS] a=approve | s=skip | e=background empty | q=quit")

        cv2.imshow("ResQVision Dataset Review", draw_boxes(image, label_path))
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("ResQVision Dataset Review")

        if key == ord("q"):
            save_manifest(manifest)
            print("[REVIEW] Progress saved.")
            return 0
        if key == ord("a"):
            copy_approved(image_path, label_path, approved_images_dir, approved_labels_dir)
            row["status"] = "approved"
        elif key == ord("e"):
            copy_approved(image_path, label_path, approved_images_dir, approved_labels_dir, background=True)
            row["status"] = "background"
        else:
            row["status"] = "skipped"

        save_manifest(manifest)

    approved = count_status(manifest, "approved")
    background = count_status(manifest, "background")
    skipped = count_status(manifest, "skipped")
    print(f"[REVIEW DONE] Approved: {approved} | Background: {background} | Skipped: {skipped}")
    print(f"Run scripts/prepare_dataset.py --dataset-root {args.dataset_root} to split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
