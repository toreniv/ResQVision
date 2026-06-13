from __future__ import annotations

import pathlib
import random
import shutil


DATASET_DIR = pathlib.Path("dataset")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_IMG_DIR = IMAGES_DIR / "train"
VAL_IMG_DIR = IMAGES_DIR / "val"
TRAIN_LBL_DIR = LABELS_DIR / "train"
VAL_LBL_DIR = LABELS_DIR / "val"
MIN_APPROVED_FRAMES = 10
RANDOM_SEED = 42


def read_label(label_path: pathlib.Path) -> str:
    if not label_path.exists():
        return ""
    return label_path.read_text(encoding="utf-8").strip()


def collect_flat_samples() -> list[tuple[pathlib.Path, pathlib.Path, bool]]:
    samples = []
    for image_path in sorted(IMAGES_DIR.glob("*.jpg")):
        label_path = LABELS_DIR / f"{image_path.stem}.txt"
        samples.append((image_path, label_path, bool(read_label(label_path))))
    return samples


def reset_split_dirs() -> None:
    for directory in (TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def choose_validation(samples: list[tuple[pathlib.Path, pathlib.Path, bool]]) -> set[str]:
    random.seed(RANDOM_SEED)
    shuffled = samples[:]
    random.shuffle(shuffled)
    val_count = max(2, len(samples) // 5)
    val_samples = shuffled[:val_count]

    has_person = any(sample[2] for sample in val_samples)
    has_background = any(not sample[2] for sample in val_samples)
    person_pool = [sample for sample in shuffled[val_count:] if sample[2]]
    background_pool = [sample for sample in shuffled[val_count:] if not sample[2]]

    if not has_person and person_pool:
        val_samples[-1] = person_pool[0]
    if not has_background and background_pool:
        replace_index = 0 if any(sample[2] for sample in val_samples[1:]) else -1
        val_samples[replace_index] = background_pool[0]

    return {sample[0].stem for sample in val_samples}


def write_data_yaml() -> None:
    yaml_content = """path: .
train: images/train
val: images/val
nc: 1
names: ['person']
"""
    (DATASET_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")


def main() -> int:
    samples = collect_flat_samples()
    if len(samples) < MIN_APPROVED_FRAMES:
        print("[ERROR] Not enough approved frames.")
        print("Review more frames with scripts/review_dataset.py")
        return 1

    val_stems = choose_validation(samples)
    reset_split_dirs()

    train_count = 0
    val_count = 0
    for image_path, label_path, _has_person in samples:
        is_val = image_path.stem in val_stems
        image_target = (VAL_IMG_DIR if is_val else TRAIN_IMG_DIR) / image_path.name
        label_target = (VAL_LBL_DIR if is_val else TRAIN_LBL_DIR) / f"{image_path.stem}.txt"
        shutil.copy2(image_path, image_target)
        if label_path.exists():
            shutil.copy2(label_path, label_target)
        else:
            label_target.write_text("", encoding="utf-8")
        if is_val:
            val_count += 1
        else:
            train_count += 1

    write_data_yaml()
    print(f"Train: {train_count} | Val: {val_count}")
    print("dataset/data.yaml created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
