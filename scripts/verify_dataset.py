from __future__ import annotations

import argparse
import json
import pathlib


FRAMES_DIR = pathlib.Path("frames")
DEDUPED_DIR = pathlib.Path("frames_deduped")
DRAFT_DIR = pathlib.Path("dataset_draft")
DEFAULT_DATASET_ROOT = pathlib.Path("dataset_approved_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an approved YOLO dataset.")
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=DEFAULT_DATASET_ROOT,
        help="Approved dataset root. Defaults to dataset_approved_v2.",
    )
    return parser.parse_args()


def count_images(path: pathlib.Path) -> int:
    return len(list(path.rglob("*.jpg"))) if path.exists() else 0


def label_has_person(label_path: pathlib.Path) -> bool:
    return label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())


def check_pairs(image_dir: pathlib.Path, label_dir: pathlib.Path, label: str) -> list[str]:
    errors = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label for {label}/{image_path.name}")
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            errors.append(f"Missing image for {label}/{label_path.name}")
    return errors


def validate_labels(labels_dir: pathlib.Path) -> tuple[int, list[str]]:
    empty_labels = 0
    errors = []
    for label_path in sorted(labels_dir.rglob("*.txt")):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels += 1
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Malformed label line in {label_path}:{line_number}: {line}")
                continue
            class_id, *values = parts
            if class_id != "0":
                errors.append(f"Invalid class id in {label_path}:{line_number}: {class_id}")
                continue
            try:
                floats = [float(value) for value in values]
            except ValueError:
                errors.append(f"Non-numeric YOLO value in {label_path}:{line_number}: {line}")
                continue
            if any(value < 0 or value > 1 for value in floats):
                errors.append(f"YOLO value outside [0, 1] in {label_path}:{line_number}: {line}")
    return empty_labels, errors


def load_manifest() -> dict:
    manifest_path = DRAFT_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def quick_stats(dataset_root: pathlib.Path, empty_label_count: int) -> dict[str, float | int]:
    manifest = load_manifest()
    approved = sum(1 for row in manifest.values() if row.get("status") == "approved")
    background = sum(1 for row in manifest.values() if row.get("status") == "background")
    train_labels = list((dataset_root / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_root / "labels" / "val").glob("*.txt"))
    approved_labels = train_labels + val_labels
    person_count = sum(len(label.read_text(encoding="utf-8").splitlines()) for label in approved_labels)
    approved_frames = len(approved_labels)
    average_persons = person_count / approved_frames if approved_frames else 0.0
    estimated_minutes = max(5, round(approved_frames / 80 * 12)) if approved_frames else 0
    return {
        "total_frames_extracted": count_images(FRAMES_DIR),
        "frames_after_dedup": count_images(DEDUPED_DIR),
        "draft_frames": count_images(DRAFT_DIR / "images"),
        "approved_flat_images": count_images(dataset_root / "images") - count_images(dataset_root / "images" / "train") - count_images(dataset_root / "images" / "val"),
        "approved_flat_labels": len(list((dataset_root / "labels").glob("*.txt"))) if (dataset_root / "labels").exists() else 0,
        "empty_label_count": empty_label_count,
        "split_approved_frames": approved_frames,
        "manifest_approved": approved,
        "background_samples": background,
        "train_count": count_images(dataset_root / "images" / "train"),
        "val_count": count_images(dataset_root / "images" / "val"),
        "average_persons_per_approved_frame": round(average_persons, 2),
        "estimated_t4_training_minutes": estimated_minutes,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    errors = []
    errors.extend(check_pairs(images_dir, labels_dir, "approved"))
    errors.extend(check_pairs(images_dir / "train", labels_dir / "train", "train"))
    errors.extend(check_pairs(images_dir / "val", labels_dir / "val", "val"))
    empty_label_count, label_errors = validate_labels(labels_dir)
    errors.extend(label_errors)

    data_yaml = dataset_root / "data.yaml"
    has_split = count_images(images_dir / "train") or count_images(images_dir / "val")
    if not data_yaml.exists():
        if has_split:
            errors.append(f"{data_yaml} missing")
    else:
        content = data_yaml.read_text(encoding="utf-8")
        for required in ("path: .", "train: images/train", "val: images/val", "nc: 1"):
            if required not in content:
                errors.append(f"data.yaml missing {required}")

    manifest = load_manifest()
    unreviewed_leaks = []
    for split in ("train", "val"):
        for image_path in (images_dir / split).glob("*.jpg"):
            if manifest.get(image_path.stem, {}).get("status") == "unreviewed":
                unreviewed_leaks.append(image_path.name)
    if unreviewed_leaks:
        errors.append(f"Unreviewed frames leaked into dataset: {', '.join(unreviewed_leaks)}")

    stats = quick_stats(dataset_root, empty_label_count)
    print(f"dataset_root: {dataset_root}")
    for key, value in stats.items():
        print(f"{key}: {value}")

    if errors:
        print("[FAIL]")
        for error in errors:
            print(f"- {error}")
        return 1

    if has_split and data_yaml.exists():
        print("[PASS] Dataset is valid and Colab-ready.")
    else:
        print("[PASS] Dataset root is valid. Run prepare_dataset.py after manual approvals to make it Colab-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
