# Requirements: pip install ultralytics opencv-python

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

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
DEFAULT_DRONE_CONF = 0.40
RETRY_DRONE_IMGSZ = 1920
RETRY_DRONE_CONF = 0.10
PERSON_CLASS_NAMES = {"person", "pedestrian", "people"}
TILE_SIZE = 640
TILE_OVERLAP = 0.50
TILE_STRIDE = int(TILE_SIZE * (1 - TILE_OVERLAP))
TILE_IMGSZ = 960
TILE_CONF = 0.25
NMS_IOU_THRESHOLD = 0.45
MODEL_DIR = pathlib.Path("models")
DRONE_MODEL_CANDIDATES = (
    MODEL_DIR / "drone_tactical_best.pt",
    pathlib.Path("yolov8s.pt"),
    pathlib.Path("yolov8n.pt"),
)


def default_drone_model() -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in DRONE_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return str(DRONE_MODEL_CANDIDATES[0])


def load_yolo_model(model_name: str) -> tuple[Any, str]:
    requested = pathlib.Path(model_name)
    candidate_paths = list(DRONE_MODEL_CANDIDATES)
    if requested in candidate_paths:
        start_index = candidate_paths.index(requested)
        candidates = [str(candidate) for candidate in candidate_paths[start_index:]]
    else:
        candidates = [model_name]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return YOLO(candidate), candidate
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No YOLO model candidates available.")


def parse_classes(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
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
        if cls_name.lower() not in PERSON_CLASS_NAMES:
            continue

        boxes.append({
            "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            "confidence": float(box.conf.item()),
            "class_id": cls_id,
        })
    return boxes


def filter_by_geometry(
    boxes: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    kept = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = h / w if w > 0 else 0
        frame_area = frame_width * frame_height
        relative_area = area / frame_area

        # person from overhead: not too big, not too small
        # relative area: 0.0003 to 0.015 of total frame
        # aspect ratio: 0.4 to 3.0 (taller than wide or square)
        if relative_area < 0.0003:
            print(f"[GEO_FILTER] DROPPED box={box['xyxy']} "
                  f"rel_area={relative_area:.5f} aspect={aspect:.2f}")
            continue  # too small - noise
        if relative_area > 0.015:
            print(f"[GEO_FILTER] DROPPED box={box['xyxy']} "
                  f"rel_area={relative_area:.5f} aspect={aspect:.2f}")
            continue  # too large - building/vehicle
        if aspect < 0.3 or aspect > 4.5:
            print(f"[GEO_FILTER] DROPPED box={box['xyxy']} "
                  f"rel_area={relative_area:.5f} aspect={aspect:.2f}")
            continue  # too wide - not a person silhouette
        kept.append(box)
    return kept


def enhance_with_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


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

    def collect_tile_boxes(source_image: Any) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for x1, y1, x2, y2 in tiles:
            tile = source_image[y1:y2, x1:x2]
            results = model(tile, imgsz=TILE_IMGSZ, conf=TILE_CONF)
            result = results[0]

            for box in extract_person_boxes(result):
                local_x1, local_y1, local_x2, local_y2 = box["xyxy"]
                collected.append({
                    **box,
                    "xyxy": [
                        max(0.0, min(frame_width, local_x1 + x1)),
                        max(0.0, min(frame_height, local_y1 + y1)),
                        max(0.0, min(frame_width, local_x2 + x1)),
                        max(0.0, min(frame_height, local_y2 + y1)),
                    ],
                })
        return collected

    remapped_boxes = collect_tile_boxes(image)
    filtered = filter_by_geometry(
        remapped_boxes, frame_width, frame_height
    )
    clahe_tiled_used = False

    if len(filtered) < 3:
        enhanced_image = enhance_with_clahe(image)
        enhanced_boxes = collect_tile_boxes(enhanced_image)
        remapped_boxes = remapped_boxes + enhanced_boxes
        filtered = filter_by_geometry(
            remapped_boxes, frame_width, frame_height
        )
        clahe_tiled_used = True

    tiled_time_ms = (time.perf_counter() - tiled_start) * 1000
    fusion_start = time.perf_counter()
    nms_boxes = global_nms(filtered, NMS_IOU_THRESHOLD)
    fusion_time_ms = (time.perf_counter() - fusion_start) * 1000

    return nms_boxes, {
        "sliced_inference_used": True,
        "tiled_inference_used": True,
        "tile_size": TILE_SIZE,
        "overlap": TILE_OVERLAP,
        "tile_imgsz": TILE_IMGSZ,
        "tile_conf": TILE_CONF,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "geometric_filter_applied": True,
        "clahe_second_pass_used": clahe_tiled_used,
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
    results = model(str(image_path), imgsz=imgsz, conf=conf)
    return results[0], {
        "model": model_name,
        "imgsz": imgsz,
        "conf": conf,
        "classes": None,
    }


def run_image_inference_attempt(
    image: np.ndarray,
    model_name: str,
    imgsz: int,
    conf: float,
    model: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    model = model or YOLO(model_name)
    results = model(image, imgsz=imgsz, conf=conf)
    return results[0], {
        "model": model_name,
        "imgsz": imgsz,
        "conf": conf,
        "classes": None,
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
    long_side = max(frame_width, frame_height)
    if imgsz == DEFAULT_DRONE_IMGSZ:
        if long_side > 1400:
            imgsz = 1536
        elif long_side > 1000:
            imgsz = 1280
        else:
            imgsz = 960
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_model_name = model_name or default_drone_model()

    attempts: list[dict[str, Any]] = []
    model, model_name = load_yolo_model(requested_model_name)
    full_frame_start = time.perf_counter()
    result, params = run_inference_attempt(image_path, model_name, imgsz, conf, classes, model=model)
    person_boxes = extract_person_boxes(result)
    person_boxes = filter_by_geometry(
        person_boxes, frame_width, frame_height
    )
    attempts.append({**params, "detections": len(person_boxes)})
    clahe_second_pass_used = False

    if len(person_boxes) < 3:
        enhanced_image = enhance_with_clahe(image)
        result, clahe_params = run_image_inference_attempt(enhanced_image, model_name, imgsz, conf, model=model)
        clahe_boxes = extract_person_boxes(result)
        clahe_boxes = filter_by_geometry(
            clahe_boxes, frame_width, frame_height
        )
        person_boxes = global_nms(person_boxes + clahe_boxes, NMS_IOU_THRESHOLD)
        attempts.append({**clahe_params, "detections": len(clahe_boxes), "preprocessing": "clahe"})
        clahe_second_pass_used = True

    retry_used = False

    if retry_if_empty and not person_boxes:
        result, params = run_inference_attempt(image_path, model_name, RETRY_DRONE_IMGSZ, RETRY_DRONE_CONF, classes, model=model)
        person_boxes = extract_person_boxes(result)
        person_boxes = filter_by_geometry(
            person_boxes, frame_width, frame_height
        )
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
        "geometric_filter_applied": True,
        "clahe_second_pass_used": clahe_second_pass_used,
        "num_tiles": 0,
        "detections_before_nms": len(person_boxes),
        "detections_after_nms": len(person_boxes),
        "tiled_time_ms": 0.0,
        "fusion_time_ms": 0.0,
    }

    tiled_boxes, tiled_metadata = run_tiled_inference(
        image, model, frame_width, frame_height
    )
    tiled_boxes = filter_by_geometry(
        tiled_boxes, frame_width, frame_height
    )
    final_boxes = global_nms(
        person_boxes + tiled_boxes, NMS_IOU_THRESHOLD
    )

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
    parser.add_argument("--model", default=None, help="YOLO model path/name. Default prefers models/drone_tactical_best.pt, then yolov8s.pt, then yolov8n.pt.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_DRONE_IMGSZ, help=f"Inference image size (default: {DEFAULT_DRONE_IMGSZ})")
    parser.add_argument("--conf", type=float, default=DEFAULT_DRONE_CONF, help=f"Confidence threshold (default: {DEFAULT_DRONE_CONF})")
    parser.add_argument("--classes", default=None, help="Comma-separated YOLO class IDs, 'all', or empty. Kept for metadata only; person filtering is name-based.")
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
