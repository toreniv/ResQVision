# Requirements: pip install ultralytics opencv-python

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import pathlib
import sys
import time
from typing import Any

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics opencv-python")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] pip install ultralytics opencv-python")
    sys.exit(1)


DEFAULT_IMAGE = "scripts/sample_input.jpg"
DEFAULT_DATA_DIR = pathlib.Path("frontend/public/data")
DEFAULT_DRONE_IMGSZ = 1280
DEFAULT_DRONE_CONF = 0.15
RETRY_DRONE_IMGSZ = 1536
RETRY_DRONE_CONF = 0.10
PERSON_CLASS_ID = 0
TILE_SIZE = 640
TILE_OVERLAP = 0.25
TILE_STRIDE = int(TILE_SIZE * (1 - TILE_OVERLAP))
TILE_IMGSZ = 960
TILE_CONF = 0.10
NMS_IOU_THRESHOLD = 0.45
PRIMARY_DRONE_MODEL = "yolov8s.pt"
FALLBACK_DRONE_MODEL = "yolov8n.pt"


def default_drone_model() -> str:
    return PRIMARY_DRONE_MODEL


def load_yolo_model(model_name: str) -> tuple[Any, str]:
    try:
        return YOLO(model_name), model_name
    except Exception:
        if model_name == PRIMARY_DRONE_MODEL:
            return YOLO(FALLBACK_DRONE_MODEL), FALLBACK_DRONE_MODEL
        raise


def parse_classes(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return [PERSON_CLASS_ID]
    if value.lower() in {"none", "all", "*"}:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_detection_payload(
    image_path: pathlib.Path,
    boxes: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    detections = []
    detection_id = 1

    for box in boxes:
        confidence = round(float(box["confidence"]), 2)
        x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
        w = x2 - x1
        h = y2 - y1
        center = [x1 + w // 2, y1 + h // 2]

        detections.append({
            "id": detection_id,
            "class": "person",
            "confidence": confidence,
            "bbox": [x1, y1, w, h],
            "center": center,
        })
        detection_id += 1

    return {
        "source": "offline_image",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "metadata": metadata,
        "detections": detections,
    }


def extract_person_boxes(result: Any) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        cls_name = result.names[cls_id]
        if cls_id != PERSON_CLASS_ID and cls_name != "person":
            continue

        boxes.append({
            "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            "confidence": float(box.conf.item()),
            "class_id": cls_id,
        })
    return boxes


def generate_tiles(frame_width: int, frame_height: int, tile_size: int = TILE_SIZE, stride: int = TILE_STRIDE) -> list[tuple[int, int, int, int]]:
    x_starts = list(range(0, max(frame_width - tile_size, 0) + 1, stride))
    y_starts = list(range(0, max(frame_height - tile_size, 0) + 1, stride))

    if not x_starts or x_starts[-1] != max(frame_width - tile_size, 0):
        x_starts.append(max(frame_width - tile_size, 0))
    if not y_starts or y_starts[-1] != max(frame_height - tile_size, 0):
        y_starts.append(max(frame_height - tile_size, 0))

    tiles = []
    for y in sorted(set(y_starts)):
        for x in sorted(set(x_starts)):
            tiles.append((x, y, min(x + tile_size, frame_width), min(y + tile_size, frame_height)))
    return tiles


def pure_python_nms(boxes: list[dict[str, Any]], iou_threshold: float = NMS_IOU_THRESHOLD) -> list[dict[str, Any]]:
    if not boxes:
        return []

    def area(box: list[float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def iou(a: list[float], b: list[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = area([x1, y1, x2, y2])
        union = area(a) + area(b) - inter
        return inter / union if union > 0 else 0.0

    remaining = sorted(boxes, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []

    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            candidate for candidate in remaining
            if iou(current["xyxy"], candidate["xyxy"]) <= iou_threshold
        ]

    return kept


def global_nms(boxes: list[dict[str, Any]], iou_threshold: float = NMS_IOU_THRESHOLD) -> list[dict[str, Any]]:
    if not boxes:
        return []

    try:
        import torch
        from torchvision.ops import nms

        tensor_boxes = torch.tensor([box["xyxy"] for box in boxes], dtype=torch.float32)
        scores = torch.tensor([box["confidence"] for box in boxes], dtype=torch.float32)
        keep_indices = nms(tensor_boxes, scores, iou_threshold).tolist()
        return [boxes[index] for index in keep_indices]
    except Exception:
        return pure_python_nms(boxes, iou_threshold)


def run_tiled_inference(image: Any, model: Any, frame_width: int, frame_height: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tiles = generate_tiles(frame_width, frame_height)
    remapped_boxes: list[dict[str, Any]] = []
    tiled_start = time.perf_counter()

    for x1, y1, x2, y2 in tiles:
        tile = image[y1:y2, x1:x2]
        results = model(tile, imgsz=TILE_IMGSZ, conf=TILE_CONF, classes=[PERSON_CLASS_ID])
        result = results[0]

        for box in extract_person_boxes(result):
            local_x1, local_y1, local_x2, local_y2 = box["xyxy"]
            remapped_boxes.append({
                **box,
                "xyxy": [
                    max(0.0, min(frame_width, local_x1 + x1)),
                    max(0.0, min(frame_height, local_y1 + y1)),
                    max(0.0, min(frame_width, local_x2 + x1)),
                    max(0.0, min(frame_height, local_y2 + y1)),
                ],
            })

    tiled_time_ms = (time.perf_counter() - tiled_start) * 1000
    fusion_start = time.perf_counter()
    nms_boxes = global_nms(remapped_boxes, NMS_IOU_THRESHOLD)
    fusion_time_ms = (time.perf_counter() - fusion_start) * 1000

    return nms_boxes, {
        "sliced_inference_used": True,
        "tiled_inference_used": True,
        "tile_size": TILE_SIZE,
        "overlap": TILE_OVERLAP,
        "tile_imgsz": TILE_IMGSZ,
        "tile_conf": TILE_CONF,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "num_tiles": len(tiles),
        "detections_before_nms": len(remapped_boxes),
        "detections_after_nms": len(nms_boxes),
        "tiled_time_ms": round(tiled_time_ms, 2),
        "fusion_time_ms": round(fusion_time_ms, 2),
    }


def draw_preview(image: Any, boxes: list[dict[str, Any]], output_path: pathlib.Path) -> None:
    annotated = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
        confidence = float(box["confidence"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (22, 163, 74), 2)
        label = f"person {confidence:.2f}"
        label_y = max(16, y1 - 6)
        cv2.putText(annotated, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (22, 163, 74), 2)
    cv2.imwrite(str(output_path), annotated)


def run_inference_attempt(
    image_path: pathlib.Path,
    model_name: str,
    imgsz: int,
    conf: float,
    classes: list[int] | None,
    model: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    model = model or YOLO(model_name)
    results = model(str(image_path), imgsz=imgsz, conf=conf, classes=classes)
    return results[0], {
        "model": model_name,
        "imgsz": imgsz,
        "conf": conf,
        "classes": classes,
    }


def run_yolo_detection(
    image_path: str | pathlib.Path,
    out_dir: str | pathlib.Path = DEFAULT_DATA_DIR,
    model_name: str | None = None,
    imgsz: int = DEFAULT_DRONE_IMGSZ,
    conf: float = DEFAULT_DRONE_CONF,
    classes: list[int] | None = None,
    retry_if_empty: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    image_path = pathlib.Path(image_path)
    out_dir = pathlib.Path(out_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if verbose:
        print(f"[ResQVision] Input: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    frame_height, frame_width = image.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_model_name = model_name or default_drone_model()
    classes = [PERSON_CLASS_ID] if classes is None else classes

    attempts: list[dict[str, Any]] = []
    model, model_name = load_yolo_model(requested_model_name)
    full_frame_start = time.perf_counter()
    result, params = run_inference_attempt(image_path, model_name, imgsz, conf, classes, model=model)
    person_boxes = extract_person_boxes(result)
    attempts.append({**params, "detections": len(person_boxes)})
    retry_used = False

    if retry_if_empty and not person_boxes:
        result, params = run_inference_attempt(image_path, model_name, RETRY_DRONE_IMGSZ, RETRY_DRONE_CONF, classes, model=model)
        person_boxes = extract_person_boxes(result)
        attempts.append({**params, "detections": len(person_boxes)})
        retry_used = True

    full_frame_time_ms = (time.perf_counter() - full_frame_start) * 1000
    tiled_metadata = {
        "sliced_inference_used": False,
        "tiled_inference_used": False,
        "tile_size": TILE_SIZE,
        "overlap": TILE_OVERLAP,
        "tile_imgsz": TILE_IMGSZ,
        "tile_conf": TILE_CONF,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "num_tiles": 0,
        "detections_before_nms": len(person_boxes),
        "detections_after_nms": len(person_boxes),
        "tiled_time_ms": 0.0,
        "fusion_time_ms": 0.0,
    }

    final_boxes = person_boxes
    full_frame_empty = len(person_boxes) == 0
    if full_frame_empty:
        final_boxes, tiled_metadata = run_tiled_inference(image, model, frame_width, frame_height)

    metadata = {
        "mode": "drone_demo",
        "model": params["model"],
        "imgsz": params["imgsz"],
        "conf": params["conf"],
        "classes": params["classes"],
        "retry_used": retry_used,
        "attempts": attempts,
        "full_frame_attempt": True,
        "final_detection_stage": "tiled" if tiled_metadata["sliced_inference_used"] else "full_frame",
        "full_frame_time_ms": round(full_frame_time_ms, 2),
        **tiled_metadata,
    }

    payload = build_detection_payload(image_path, final_boxes, frame_width, frame_height, metadata)

    json_path = out_dir / "detections.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    preview_path = out_dir / "detection_preview.jpg"
    draw_preview(image, final_boxes, preview_path)

    if verbose:
        detections = payload["detections"]
        if detections:
            print(f"[ResQVision] Detected {len(detections)} person(s)")
        else:
            print("[ResQVision] No persons detected. JSON saved as empty array.")
        print(f"[ResQVision] Saved -> {json_path}")
        print(f"[ResQVision] Saved -> {preview_path}")

    return {
        "payload": payload,
        "detections_path": json_path,
        "preview_path": preview_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8n on an image and export person detections for ResQVision."
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Path to input image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument("--model", default=None, help="YOLO model path/name. Default prefers local yolov8s.pt, then yolov8n.pt.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_DRONE_IMGSZ, help=f"Inference image size (default: {DEFAULT_DRONE_IMGSZ})")
    parser.add_argument("--conf", type=float, default=DEFAULT_DRONE_CONF, help=f"Confidence threshold (default: {DEFAULT_DRONE_CONF})")
    parser.add_argument("--classes", default=str(PERSON_CLASS_ID), help="Comma-separated YOLO class IDs, 'all', or empty. Default: person only.")
    parser.add_argument("--no-retry", action="store_true", help="Disable empty-result retry with larger image size and lower confidence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_yolo_detection(
            args.image,
            model_name=args.model,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=parse_classes(args.classes),
            retry_if_empty=not args.no_retry,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
