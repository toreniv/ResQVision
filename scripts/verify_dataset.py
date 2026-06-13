from __future__ import annotations

import json
import pathlib


FRAMES_DIR = pathlib.Path("frames")
DEDUPED_DIR = pathlib.Path("frames_deduped")
DRAFT_DIR = pathlib.Path("dataset_draft")
DATASET_DIR = pathlib.Path("dataset")


def count_images(path: pathlib.Path) -> int:
    return len(list(path.rglob("*.jpg"))) if path.exists() else 0


def label_has_person(label_path: pathlib.Path) -> bool:
    return label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())


def check_pairs(split: str) -> list[str]:
    errors = []
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label for {split}/{image_path.name}")
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            errors.append(f"Missing image for {split}/{label_path.name}")
    return errors


def load_manifest() -> dict:
    manifest_path = DRAFT_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def quick_stats() -> dict[str, float | int]:
    manifest = load_manifest()
    approved = sum(1 for row in manifest.values() if row.get("status") == "approved")
    background = sum(1 for row in manifest.values() if row.get("status") == "background")
    train_labels = list((DATASET_DIR / "labels" / "train").glob("*.txt"))
    val_labels = list((DATASET_DIR / "labels" / "val").glob("*.txt"))
    approved_labels = train_labels + val_labels
    person_count = sum(len(label.read_text(encoding="utf-8").splitlines()) for label in approved_labels)
    approved_frames = len(approved_labels)
    average_persons = person_count / approved_frames if approved_frames else 0.0
    estimated_minutes = max(5, round(approved_frames / 80 * 12)) if approved_frames else 0
    return {
        "total_frames_extracted": count_images(FRAMES_DIR),
        "frames_after_dedup": count_images(DEDUPED_DIR),
        "draft_frames": count_images(DRAFT_DIR / "images"),
        "approved_frames": approved_frames,
        "manifest_approved": approved,
        "background_samples": background,
        "train_count": count_images(DATASET_DIR / "images" / "train"),
        "val_count": count_images(DATASET_DIR / "images" / "val"),
        "average_persons_per_approved_frame": round(average_persons, 2),
        "estimated_t4_training_minutes": estimated_minutes,
    }


def main() -> int:
    errors = []
    errors.extend(check_pairs("train"))
    errors.extend(check_pairs("val"))

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        errors.append("dataset/data.yaml missing")
    else:
        content = data_yaml.read_text(encoding="utf-8")
        for required in ("path: .", "train: images/train", "val: images/val", "nc: 1"):
            if required not in content:
                errors.append(f"data.yaml missing {required}")

    manifest = load_manifest()
    unreviewed_leaks = []
    for split in ("train", "val"):
        for image_path in (DATASET_DIR / "images" / split).glob("*.jpg"):
            if manifest.get(image_path.stem, {}).get("status") == "unreviewed":
                unreviewed_leaks.append(image_path.name)
    if unreviewed_leaks:
        errors.append(f"Unreviewed frames leaked into dataset: {', '.join(unreviewed_leaks)}")

    stats = quick_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    if errors:
        print("[FAIL]")
        for error in errors:
            print(f"- {error}")
        return 1

    print("[PASS] Dataset is valid and Colab-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
