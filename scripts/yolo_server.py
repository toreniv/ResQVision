from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("[ERROR] pip install fastapi uvicorn python-multipart")
    raise SystemExit(1)

import cv2

from fuse_yolo_to_tactical import FUSION_PATH, main as run_tactical_fusion
from yolo_detect import DEFAULT_DATA_DIR, run_yolo_detection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"
UPLOAD_IMAGE_PATH = TEMP_UPLOAD_DIR / "drone_frame.png"
MANUAL_MARKERS_PATH = DEFAULT_DATA_DIR / "manual_markers.json"
DATASET_BUILDER_DIR = TEMP_UPLOAD_DIR / "dataset_builder"
DATASET_BUILDER_IMAGES_DIR = DATASET_BUILDER_DIR / "images"
DATASET_BUILDER_LABELS_DIR = DATASET_BUILDER_DIR / "labels"
DATASET_ZIP_PATH = PROJECT_ROOT / "dataset.zip"
DEFAULT_BOX_WIDTH = 40
DEFAULT_BOX_HEIGHT = 80

app = FastAPI(title="ResQVision Local YOLO Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_json(path: Path, fallback: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def next_dataset_frame_id() -> str:
    DATASET_BUILDER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    existing_numbers = []
    for image_path in DATASET_BUILDER_IMAGES_DIR.glob("frame_*.jpg"):
        try:
            existing_numbers.append(int(image_path.stem.replace("frame_", "")))
        except ValueError:
            continue
    return f"frame_{(max(existing_numbers, default=0) + 1):04d}"


def marker_to_yolo_label(marker: dict[str, Any], frame_width: int, frame_height: int) -> str | None:
    image_center = marker.get("image_center")
    x_image = marker.get("x_image")
    y_image = marker.get("y_image")
    if (x_image is None or y_image is None) and isinstance(image_center, list) and len(image_center) >= 2:
        x_image = image_center[0]
        y_image = image_center[1]
    if x_image is None or y_image is None:
        return None

    center_x = float(x_image)
    center_y = float(y_image)
    half_w = DEFAULT_BOX_WIDTH / 2
    half_h = DEFAULT_BOX_HEIGHT / 2
    x1 = max(0.0, center_x - half_w)
    y1 = max(0.0, center_y - half_h)
    x2 = min(float(frame_width), center_x + half_w)
    y2 = min(float(frame_height), center_y + half_h)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    yolo_center_x = ((x1 + x2) / 2) / frame_width
    yolo_center_y = ((y1 + y2) / 2) / frame_height
    yolo_w = box_w / frame_width
    yolo_h = box_h / frame_height
    return f"0 {yolo_center_x:.6f} {yolo_center_y:.6f} {yolo_w:.6f} {yolo_h:.6f}"


def save_dataset_builder_sample(markers: list[dict[str, Any]]) -> dict[str, Any]:
    if not markers:
        return {"created": False, "labels": 0, "reason": "no_markers"}
    if not UPLOAD_IMAGE_PATH.exists():
        return {"created": False, "labels": 0, "reason": "missing_uploaded_image"}

    image = cv2.imread(str(UPLOAD_IMAGE_PATH))
    if image is None:
        return {"created": False, "labels": 0, "reason": "image_decode_failed"}

    frame_height, frame_width = image.shape[:2]
    label_lines = [
        label for marker in markers
        if (label := marker_to_yolo_label(marker, frame_width, frame_height)) is not None
    ]
    if not label_lines:
        return {"created": False, "labels": 0, "reason": "no_valid_labels"}

    frame_id = next_dataset_frame_id()
    DATASET_BUILDER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_BUILDER_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = DATASET_BUILDER_IMAGES_DIR / f"{frame_id}.jpg"
    label_path = DATASET_BUILDER_LABELS_DIR / f"{frame_id}.txt"
    cv2.imwrite(str(image_path), image)
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return {"created": True, "frame": frame_id, "labels": len(label_lines)}


def export_dataset_zip() -> dict[str, Any]:
    image_paths = sorted(DATASET_BUILDER_IMAGES_DIR.glob("frame_*.jpg"))
    paired_samples = [
        (image_path, DATASET_BUILDER_LABELS_DIR / f"{image_path.stem}.txt")
        for image_path in image_paths
        if (DATASET_BUILDER_LABELS_DIR / f"{image_path.stem}.txt").exists()
    ]

    export_root = TEMP_UPLOAD_DIR / "dataset_export"
    dataset_root = export_root / "dataset"
    if export_root.exists():
        shutil.rmtree(export_root)

    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    total = len(paired_samples)
    if total == 0:
        return {
            "ok": False,
            "message": "No dataset builder samples found. Save tactical tags after uploading an image first.",
            "images": 0,
            "labels": 0,
            "zip_path": DATASET_ZIP_PATH.name,
        }

    val_count = max(1, round(total * 0.2)) if total > 1 else 0
    train_count = total - val_count

    for index, (image_path, label_path) in enumerate(paired_samples):
        split = "train" if index < train_count else "val"
        shutil.copy2(image_path, dataset_root / "images" / split / image_path.name)
        shutil.copy2(label_path, dataset_root / "labels" / split / label_path.name)

    (dataset_root / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  - person\n",
        encoding="utf-8",
    )

    if DATASET_ZIP_PATH.exists():
        DATASET_ZIP_PATH.unlink()
    with zipfile.ZipFile(DATASET_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(dataset_root.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(dataset_root).as_posix())

    return {
        "ok": True,
        "images": total,
        "labels": total,
        "zip_path": DATASET_ZIP_PATH.name,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "resqvision-yolo-local"}


@app.post("/api/yolo/upload")
async def upload_yolo_image(image: UploadFile = File(...)) -> dict[str, Any]:
    try:
        TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with UPLOAD_IMAGE_PATH.open("wb") as handle:
            shutil.copyfileobj(image.file, handle)

        result = run_yolo_detection(UPLOAD_IMAGE_PATH, DEFAULT_DATA_DIR, verbose=False)
        run_tactical_fusion()
        fusion = load_json(FUSION_PATH, {})

        return {
            "ok": True,
            "detections": len(result["payload"].get("detections", [])),
            "fusion_mode": fusion.get("fusion_mode", "YOLO_IMAGE"),
            "metadata": result["payload"].get("metadata", {}),
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


@app.post("/api/markers/save")
async def save_manual_markers(markers: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        save_json(MANUAL_MARKERS_PATH, markers)
        dataset_sample = save_dataset_builder_sample(markers)
        run_tactical_fusion()
        fusion = load_json(FUSION_PATH, {})
        return {
            "ok": True,
            "markers": len(markers),
            "dataset_sample": dataset_sample,
            "fusion_mode": fusion.get("fusion_mode", "YOLO_FUSION"),
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


@app.post("/api/dataset/export")
async def export_training_dataset() -> dict[str, Any]:
    try:
        return export_dataset_zip()
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


@app.get("/api/dataset/download")
async def download_training_dataset() -> FileResponse:
    return FileResponse(DATASET_ZIP_PATH, filename=DATASET_ZIP_PATH.name)


def main() -> int:
    uvicorn.run(app, host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
