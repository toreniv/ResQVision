from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("[ERROR] pip install fastapi uvicorn python-multipart")
    raise SystemExit(1)

from fuse_yolo_to_tactical import FUSION_PATH, main as run_tactical_fusion
from yolo_detect import DEFAULT_DATA_DIR, run_yolo_detection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"
UPLOAD_IMAGE_PATH = TEMP_UPLOAD_DIR / "drone_frame.png"
MANUAL_MARKERS_PATH = DEFAULT_DATA_DIR / "manual_markers.json"

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
        run_tactical_fusion()
        fusion = load_json(FUSION_PATH, {})
        return {
            "ok": True,
            "markers": len(markers),
            "fusion_mode": fusion.get("fusion_mode", "YOLO_FUSION"),
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def main() -> int:
    uvicorn.run(app, host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
