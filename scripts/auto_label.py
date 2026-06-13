from __future__ import annotations

import json
import pathlib
import shutil

import cv2

# Import run_yolo_detection from yolo_detect
from yolo_detect import run_yolo_detection


FRAMES_DIR = pathlib.Path("frames_deduped")
DRAFT_DIR = pathlib.Path("dataset_draft")
IMAGES_DIR = DRAFT_DIR / "images"
LABELS_DIR = DRAFT_DIR / "labels"
DETAILS_DIR = DRAFT_DIR / "details"
MANIFEST_PATH = DRAFT_DIR / "manifest.json"

MANUAL_GT_PATH = pathlib.Path("benchmarks/manual_expected_soldiers.json")


def main() -> int:
    if DRAFT_DIR.exists():
        shutil.rmtree(DRAFT_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    
    expected_gt = {}
    if MANUAL_GT_PATH.exists():
        try:
            expected_gt = json.loads(MANUAL_GT_PATH.read_text("utf-8"))
        except:
            pass

    for img_path in sorted(FRAMES_DIR.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        height, width = img.shape[:2]
        
        img_details_dir = DETAILS_DIR / img_path.stem
        img_details_dir.mkdir(parents=True, exist_ok=True)
        
        out = run_yolo_detection(
            image_path=str(img_path),
            out_dir=img_details_dir,
            model_name=None,
            tile_size=480,
            tile_imgsz=1280,
            overlap_ratio=0.35,
            nms_iou=0.65,
            verbose=False
        )
        
        detections = out["payload"]["detections"]
        detected_count = len(detections)
        
        label_lines = []
        for det in detections:
            x1, y1, w, h = det["bbox"]
            x2 = x1 + w
            y2 = y1 + h
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
        shutil.copy2(img_path, IMAGES_DIR / img_path.name)
        (LABELS_DIR / img_path.with_suffix(".txt").name).write_text("\n".join(label_lines), encoding="utf-8")
        
        expected_count = expected_gt.get(img_path.name, None)
        
        incomplete_labels = True if (expected_count is not None and detected_count < expected_count) else False
        count_ratio = None
        if expected_count is not None and expected_count > 0:
            count_ratio = (detected_count / expected_count) * 100

        manifest[img_path.stem] = {
            "filename": img_path.name,
            "preset_used": "high_recall",
            "detected_count": detected_count,
            "expected_count": expected_count,
            "count_ratio": count_ratio,
            "needs_review": True,
            "incomplete_labels": incomplete_labels,
            "status": "unreviewed",
        }
        
        shutil.copy2(img_path, img_details_dir / img_path.name)
        (img_details_dir / img_path.with_suffix(".txt").name).write_text("\n".join(label_lines), encoding="utf-8")
        
        print(f"{img_path.name}: {detected_count} draft person(s) (high_recall)")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[DRAFT] These labels are unreviewed.")
    print("Do not use dataset_draft/ for training directly.")
    print("Run scripts/review_dataset.py to approve frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
