from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any
import uuid
import zipfile

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
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


def save_dataset_builder_sample_from_image(
    image: Any,
    markers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Like save_dataset_builder_sample but operates on an already-decoded ndarray.

    This avoids the race condition where drone_frame.png is overwritten by the
    next upload before save_dataset_builder_sample can read it.
    """
    if image is None:
        return {"created": False, "labels": 0, "reason": "image_none"}
    if not markers:
        return {"created": False, "labels": 0, "reason": "no_markers"}

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

    # Always guarantee at least 1 sample in val so the training script passes
    # dataset validation. For >= 5 samples use an 80/20 split; for smaller
    # datasets donate 1 sample to val regardless of how few remain for train.
    if total >= 5:
        val_count = max(1, round(total * 0.2))
    else:
        val_count = 1
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
        "  - soldier\n",
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


class BenchmarkData(BaseModel):
    image_filename: str
    expected_soldiers: int
    detected_soldiers: int
    estimated_recall: float
    model_name: str
    tile_size: int
    overlap_ratio: float
    min_rel_area: float

BENCHMARKS_JSON_PATH = PROJECT_ROOT / "frontend" / "public" / "data" / "benchmarks.json"

@app.post("/api/benchmark/save")
async def save_benchmark(data: BenchmarkData) -> dict[str, Any]:
    try:
        benchmarks = []
        if BENCHMARKS_JSON_PATH.exists():
            with BENCHMARKS_JSON_PATH.open("r", encoding="utf-8") as f:
                try:
                    benchmarks = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        # Avoid duplicate exact records
        record = data.dict()
        if not any(b == record for b in benchmarks):
            benchmarks.append(record)
        
            with BENCHMARKS_JSON_PATH.open("w", encoding="utf-8") as f:
                json.dump(benchmarks, f, indent=2)
            
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


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


@app.post("/api/yolo/upload_and_save")
async def upload_and_save_dataset_sample(
    image: UploadFile = File(...),
    manual_markers: str = "",
) -> dict[str, Any]:
    """Atomic endpoint for batch processing.

    1. Saves the uploaded image.
    2. Runs YOLO inference.
    3. Converts each YOLO detection bbox to a marker using real pixel coords.
    4. Merges with any manual_markers supplied as a JSON string.
    5. Writes one dataset sample before returning — avoids the race condition
       where drone_frame.png is overwritten by the next batch upload.
    """
    try:
        TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Read the raw bytes before writing so we keep the image in memory.
        raw_bytes = await image.read()

        # Persist to drone_frame.png (keeps existing single-image workflow intact).
        UPLOAD_IMAGE_PATH.write_bytes(raw_bytes)

        # Decode into a cv2 ndarray — our ground truth for this sample.
        import numpy as np
        nparr = np.frombuffer(raw_bytes, np.uint8)
        cv2_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv2_image is None:
            return {"ok": False, "message": "Could not decode uploaded image"}

        frame_height, frame_width = cv2_image.shape[:2]

        # Run YOLO on the persisted file.
        result = run_yolo_detection(UPLOAD_IMAGE_PATH, DEFAULT_DATA_DIR, verbose=False)
        run_tactical_fusion()
        fusion = load_json(FUSION_PATH, {})

        yolo_detections = result["payload"].get("detections", [])

        # Build markers from YOLO detections using real bbox pixel coords.
        # bbox format from yolo_detect.py: [x1, y1, w, h]
        yolo_markers: list[dict[str, Any]] = []
        for det in yolo_detections:
            bbox = det.get("bbox")  # [x1, y1, w, h]
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, w, h = bbox
            cx = x1 + w / 2
            cy = y1 + h / 2
            # Store as a full YOLO label line directly (bypass marker_to_yolo_label
            # default 40x80 approximation — use the real detected box instead).
            cx_norm = cx / frame_width
            cy_norm = cy / frame_height
            w_norm  = w  / frame_width
            h_norm  = h  / frame_height
            yolo_markers.append({
                "_yolo_label": f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}",
                "x_image": cx,
                "y_image": cy,
                "source": "YOLO_AUTO",
            })

        # Parse and merge any manual markers from the request.
        extra_markers: list[dict[str, Any]] = []
        if manual_markers.strip():
            try:
                parsed = json.loads(manual_markers)
                if isinstance(parsed, list):
                    extra_markers = parsed
            except json.JSONDecodeError:
                pass

        all_markers = yolo_markers + extra_markers

        # Always write a dataset sample — even if zero detections.
        # An empty label file is valid YOLO (negative sample / background).
        # The frontend flags zero-detection images as "needs_review".
        frame_id = next_dataset_frame_id()
        DATASET_BUILDER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_BUILDER_LABELS_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DATASET_BUILDER_IMAGES_DIR / f"{frame_id}.jpg"), cv2_image)

        # Build label lines (empty list = empty label file = valid YOLO negative sample).
        label_lines: list[str] = []
        for marker in all_markers:
            if "_yolo_label" in marker:
                label_lines.append(marker["_yolo_label"])
            else:
                line = marker_to_yolo_label(marker, frame_width, frame_height)
                if line:
                    label_lines.append(line)

        label_content = ("\n".join(label_lines) + "\n") if label_lines else ""
        (DATASET_BUILDER_LABELS_DIR / f"{frame_id}.txt").write_text(label_content, encoding="utf-8")

        dataset_sample = {
            "created": True,
            "frame": frame_id,
            "labels": len(label_lines),
            "yolo_boxes": len(yolo_markers),
            "manual_boxes": len(extra_markers),
            "needs_review": len(label_lines) == 0,
        }

        return {
            "ok": True,
            "detections": len(yolo_detections),
            "fusion_mode": fusion.get("fusion_mode", "YOLO_IMAGE"),
            "metadata": result["payload"].get("metadata", {}),
            "dataset_sample": dataset_sample,
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


# ---------------------------------------------------------------------------
# Pipeline: Build & Train
# ---------------------------------------------------------------------------

TRAINING_JOBS: dict[str, dict[str, Any]] = {}

def _reader_thread(process: subprocess.Popen, job_id: str) -> None:
    """Reads stdout from the process and appends to the job's log deque safely."""
    try:
        if process.stdout:
            for line in iter(process.stdout.readline, b""):
                line_str = line.decode("utf-8", errors="replace").rstrip("\r\n")
                TRAINING_JOBS[job_id]["log"].append(line_str)
    except Exception as exc:
        TRAINING_JOBS[job_id]["log"].append(f"[Reader thread error: {exc}]")
    finally:
        process.wait()

@app.post("/api/pipeline/build-and-train")
async def pipeline_build_and_train(payload: dict[str, Any]) -> dict[str, Any]:
    """1. Exports dataset.zip 2. Spawns train_tactical.py in background."""
    try:
        epochs = int(payload.get("epochs", 3))
        batch = int(payload.get("batch", 4))

        # 1. Export dataset synchronously (usually fast)
        export_result = export_dataset_zip()
        if not export_result.get("ok"):
            return export_result

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if not cuda_available:
            return {
                "ok": True,
                "status": "local_training_skipped_colab_required",
                "local_gpu_available": False,
                "dataset_zip_path": export_result.get("zip_path", "dataset.zip"),
                "message": "No local GPU detected. Dataset is ready for Colab training.",
                "dataset": export_result,
            }

        # 2. Launch training script as background subprocess
        train_script = PROJECT_ROOT / "scripts" / "train_tactical.py"
        cmd = [sys.executable, str(train_script), "--epochs", str(epochs), "--batch", str(batch)]

        job_id = uuid.uuid4().hex[:8]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,  # Line buffered
        )

        TRAINING_JOBS[job_id] = {
            "process": process,
            "log": deque(maxlen=2000),  # Keep last 2000 lines max in memory
        }

        # Start background reader thread (Windows safe, avoids select() on pipes)
        thread = threading.Thread(target=_reader_thread, args=(process, job_id), daemon=True)
        thread.start()

        return {
            "ok": True,
            "job_id": job_id,
            "dataset": export_result,
            "message": "Training started in background.",
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}

@app.get("/api/pipeline/status/{job_id}")
async def pipeline_status(job_id: str) -> dict[str, Any]:
    """Returns the current state of a training job + log tail."""
    job = TRAINING_JOBS.get(job_id)
    if not job:
        return {"ok": False, "message": "Job not found"}

    process: subprocess.Popen = job["process"]
    # We consume lines from the deque so the frontend only gets the delta each poll.
    # We pop from the left to read chronologically.
    log_tail: list[str] = []
    log_deque: deque[str] = job["log"]
    while log_deque:
        try:
            log_tail.append(log_deque.popleft())
        except IndexError:
            break

    returncode = process.poll()
    running = returncode is None

    return {
        "ok": True,
        "running": running,
        "returncode": returncode,
        "log_tail": log_tail,
    }

def main() -> int:
    uvicorn.run(app, host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
