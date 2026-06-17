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
CONFIRMED_MIN_CONF = 0.30
CANDIDATE_MIN_CONF = 0.08
RETRY_DRONE_IMGSZ = 1920
RETRY_DRONE_CONF = 0.10
PERSON_CLASS_NAMES = {"person", "pedestrian", "people", "soldier"}
TILE_SIZE = 640
TILE_OVERLAP = 0.50
TILE_STRIDE = int(TILE_SIZE * (1 - TILE_OVERLAP))
TILE_IMGSZ = 960
TILE_CONF = 0.25
NMS_IOU_THRESHOLD = 0.45
MODEL_DIR = pathlib.Path("models")
DRONE_MODEL_CANDIDATES = (
    MODEL_DIR / "drone_tactical_best.pt",
    pathlib.Path("mshamrai/yolov8m-visdrone"),
    pathlib.Path("mshamrai/yolov8s-visdrone"),
    # Optional upgrade path for higher recall but slower inference:
    # pathlib.Path("mshamrai/yolov8l-visdrone"),
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
    for idx, candidate in enumerate(candidates):
        try:
            model = YOLO(candidate)
            device = model.device.type if hasattr(model, "device") else "unknown"
            source = "local" if pathlib.Path(candidate).exists() else "Hugging Face / remote"
            is_fallback = idx > 0
            print(f"[MODEL] Loaded: {candidate}")
            print(f"[MODEL] Source: {source} | Device: {device} | Fallback triggered: {'Yes' if is_fallback else 'No'}")
            return model, candidate
        except Exception as exc:
            print(f"[MODEL] Failed to load {candidate}: {exc}")
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


def apply_confidence_and_shadow_policy(
    image, boxes, frame_width, frame_height, metadata
):
    processed = []
    detection_id = 1
    
    import time

    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        interArea = max(0, x2 - x1) * max(0, y2 - y1)
        if interArea == 0: return 0.0
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return interArea / float(max(box1Area + box2Area - interArea, 1))

    for box in boxes:
        conf = float(box["confidence"])
        x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
        
        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))
        
        w = x2 - x1
        h = y2 - y1
        
        det = {
            "id": f"det_{detection_id:03d}",
            "class_name": "person",
            "class": "person",
            "confidence": round(conf, 2),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [x1, y1, w, h],
            "bbox": [x1, y1, w, h],
            "center": [x1 + w // 2, y1 + h // 2],
            "source": "YOLO",
            "model": metadata.get("model", "unknown"),
            "stage": metadata.get("final_detection_stage", "hybrid"),
        }
        detection_id += 1

        if w <= 0 or h <= 0:
            det["status"] = "rejected_shadow_candidate"
            det["reason"] = "invalid_box_geometry"
            det["review_required"] = False
            processed.append(det)
            continue
            
        area = w * h
        rel_area = area / float(max(frame_width * frame_height, 1))
        aspect = w / float(h)
        
        min_aspect = metadata.get("min_aspect", 0.25)
        max_aspect = metadata.get("max_aspect", 3.0)
        min_rel_area = metadata.get("min_rel_area", 0.00005)
        max_rel_area = metadata.get("max_rel_area", 0.08)

        det["aspect_width_over_height"] = round(aspect, 3)
        det["relative_area"] = round(rel_area, 6)

        if conf < CANDIDATE_MIN_CONF:
            det["status"] = "rejected_low_confidence"
            det["reason"] = "low_confidence"
            det["review_required"] = False
        elif aspect < min_aspect or aspect > max_aspect:
            det["status"] = "rejected_shadow_candidate"
            det["reason"] = "extreme_aspect_ratio"
            det["review_required"] = False
        elif rel_area < min_rel_area or rel_area > max_rel_area:
            det["status"] = "rejected_shadow_candidate"
            det["reason"] = "abnormal_relative_area"
            det["review_required"] = False
        elif conf < CONFIRMED_MIN_CONF:
            if 0.4 <= aspect <= 3.0:
                det["status"] = "candidate"
                det["candidate_strength"] = "strong" if 0.6 <= aspect <= 2.0 else "weak"
                det["reason"] = "low_conf_valid_top_down_geometry"
                det["review_required"] = True
            else:
                det["status"] = "rejected_shadow_candidate"
                det["reason"] = "extreme_aspect_ratio"
                det["review_required"] = False
        else:
            det["status"] = "confirmed"
            det["reason"] = ""
            det["review_required"] = False

        processed.append(det)

    confirmed_boxes = [d for d in processed if d["status"] == "confirmed"]
    for det in processed:
        if det["status"] == "candidate":
            for c_det in confirmed_boxes:
                iou = compute_iou(det["bbox_xyxy"], c_det["bbox_xyxy"])
                if iou > 0.15:
                    det["status"] = "rejected_shadow_candidate"
                    det["reason"] = "adjacent_shadow_noise"
                    det["review_required"] = False
                    break

    import cv2
    import pathlib
    if image is not None:
        negatives_dir = pathlib.Path("data/negatives_for_training")
        for det in processed:
            if det["status"] == "rejected_shadow_candidate":
                negatives_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                import uuid
                uid = uuid.uuid4().hex[:6]
                fname = negatives_dir / f"shadow_{ts}_{uid}.jpg"
                x1, y1, x2, y2 = det["bbox_xyxy"]
                if x2 > x1 and y2 > y1:
                    try:
                        crop = image[y1:y2, x1:x2]
                        if crop.size > 0:
                            cv2.imwrite(str(fname), crop)
                            print(f"[GEO_FILTER] Saved hard negative -> {fname}")
                    except Exception as e:
                        pass

    counts = {
        "confirmed_count": sum(1 for d in processed if d["status"] == "confirmed"),
        "candidate_count": sum(1 for d in processed if d["status"] == "candidate"),
        "rejected_count": sum(1 for d in processed if d["status"].startswith("rejected_")),
    }
    
    return processed, counts

def build_detection_payload(
    image,
    image_path,
    boxes,
    frame_width,
    frame_height,
    metadata,
):
    from datetime import datetime, timezone
    processed_detections, counts = apply_confidence_and_shadow_policy(
        image, boxes, frame_width, frame_height, metadata
    )
    
    top_level_metadata = {
        **metadata,
        "confirmed_count": counts["confirmed_count"],
        "candidate_count": counts["candidate_count"],
        "rejected_count": counts["rejected_count"],
        "review_required": counts["candidate_count"] > 0 and counts["confirmed_count"] == 0,
        "confirmed_min_conf": CONFIRMED_MIN_CONF,
        "candidate_min_conf": CANDIDATE_MIN_CONF,
        "confidence_policy": "confirmed_only_for_tactical_fusion",
    }

    return {
        "source": "offline_image",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "metadata": top_level_metadata,
        "detections": processed_detections,
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
    boxes,
    frame_width,
    frame_height,
    min_rel_area=0.00005,
    max_rel_area=0.08,
    min_aspect=0.25,
    max_aspect=3.0,
):
    print(f"[GEO_FILTER] Passthrough: {len(boxes)} boxes passed to policy")
    return boxes

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


def run_tiled_inference(
    image: Any, model: Any, frame_width: int, frame_height: int, 
    min_rel_area: float = 0.00005, max_rel_area: float = 0.08,
    min_aspect: float = 0.25, max_aspect: float = 3.0,
    geo_filter_enabled: bool = True, tile_size: int = 640, overlap_ratio: float = 0.35,
    tile_imgsz: int = 960, verbose: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stride = int(tile_size * (1 - overlap_ratio))
    tiles = generate_tiles(frame_width, frame_height, tile_size, stride)
    
    tiled_raw_total = 0
    after_class_filter = 0
    detections_per_tile = []
    
    tiled_start = time.perf_counter()

    def collect_tile_boxes(source_image: Any) -> list[dict[str, Any]]:
        nonlocal tiled_raw_total, after_class_filter
        collected: list[dict[str, Any]] = []
        for i, (x1, y1, x2, y2) in enumerate(tiles):
            tile = source_image[y1:y2, x1:x2]
            results = model(tile, imgsz=tile_imgsz, conf=TILE_CONF)
            result = results[0]
            
            tiled_raw_total += len(result.boxes)
            person_boxes = extract_person_boxes(result)
            after_class_filter += len(person_boxes)
            
            detections_per_tile.append(len(person_boxes))

            for box in person_boxes:
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
    if geo_filter_enabled:
        filtered = filter_by_geometry(
            remapped_boxes,
            frame_width,
            frame_height,
            min_rel_area,
            max_rel_area,
            min_aspect,
            max_aspect,
        )
    else:
        filtered = remapped_boxes
    after_geometry_filter = len(filtered)
    
    tiled_time_ms = (time.perf_counter() - tiled_start) * 1000

    return filtered, {
        "num_tiles": len(tiles),
        "detections_per_tile": detections_per_tile,
        "tiled_raw_total": tiled_raw_total,
        "after_class_filter_tiled": after_class_filter,
        "after_geometry_filter_tiled": after_geometry_filter,
        "tiled_time_ms": round(tiled_time_ms, 2)
    }

def draw_preview(image, boxes, output_path) -> None:
    annotated = image.copy()
    
    import cv2
    cv2.putText(annotated, "[PREVIEW] confidence_policy_v2_active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    for box in boxes:
        status = box.get("status", "confirmed")
        if status.startswith("rejected_"):
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box.get("bbox_xyxy", box.get("xyxy", [0,0,0,0]))]
        confidence = float(box["confidence"])
        
        if status == "candidate":
            color = (0, 215, 255)
            label = f"candidate person {confidence:.2f}"
        else:
            color = (22, 163, 74)
            label = f"person {confidence:.2f}"
            
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label_y = max(16, y1 - 6)
        cv2.putText(annotated, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
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
    min_rel_area: float = 0.00005,
    max_rel_area: float = 0.08,
    min_aspect: float = 0.25,
    max_aspect: float = 3.0,
    geo_filter_enabled: bool = True,
    preset: str = "fast",
    tile_size: int = 640,
    overlap_ratio: float = 0.35,
    tile_imgsz: int = 960,
    nms_iou: float = 0.65,
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
    if imgsz == 1280:
        if long_side > 1400:
            imgsz = 1536
        elif long_side > 1000:
            imgsz = 1280
        else:
            imgsz = 960
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_model_name = model_name or default_drone_model()

    model, model_name_actual = load_yolo_model(requested_model_name)
    
    attempts = []
    
    # 1. Full Frame Pass
    full_frame_start = time.perf_counter()
    result, params = run_inference_attempt(image_path, model_name_actual, imgsz, conf, classes, model=model)
    full_frame_raw = len(result.boxes)
    full_frame_person_boxes = extract_person_boxes(result)
    after_class_filter_ff = len(full_frame_person_boxes)
    if geo_filter_enabled:
        person_boxes = filter_by_geometry(
            full_frame_person_boxes,
            frame_width,
            frame_height,
            min_rel_area,
            max_rel_area,
            min_aspect,
            max_aspect,
        )
    else:
        person_boxes = full_frame_person_boxes
    after_geometry_filter_ff = len(person_boxes)
    attempts.append({**params, "detections": len(person_boxes)})
    
    # 2. CLAHE Pass if detections < 3
    clahe_raw = 0
    after_class_filter_clahe = 0
    clahe_boxes = []
    if len(person_boxes) < 3:
        enhanced_image = enhance_with_clahe(image)
        c_result, clahe_params = run_image_inference_attempt(enhanced_image, model_name_actual, imgsz, conf, model=model)
        clahe_raw = len(c_result.boxes)
        c_boxes = extract_person_boxes(c_result)
        after_class_filter_clahe = len(c_boxes)
        if geo_filter_enabled:
            clahe_boxes = filter_by_geometry(
                c_boxes,
                frame_width,
                frame_height,
                min_rel_area,
                max_rel_area,
                min_aspect,
                max_aspect,
            )
        else:
            clahe_boxes = c_boxes
        attempts.append({**clahe_params, "detections": len(clahe_boxes), "preprocessing": "clahe"})

    # 3. High-res Retry if empty
    high_res_retry_raw = 0
    after_class_filter_retry = 0
    retry_boxes = []
    retry_used = False
    if retry_if_empty and not person_boxes and not clahe_boxes:
        hr_result, hr_params = run_inference_attempt(image_path, model_name_actual, RETRY_DRONE_IMGSZ, RETRY_DRONE_CONF, classes, model=model)
        high_res_retry_raw = len(hr_result.boxes)
        r_boxes = extract_person_boxes(hr_result)
        after_class_filter_retry = len(r_boxes)
        if geo_filter_enabled:
            retry_boxes = filter_by_geometry(
                r_boxes,
                frame_width,
                frame_height,
                min_rel_area,
                max_rel_area,
                min_aspect,
                max_aspect,
            )
        else:
            retry_boxes = r_boxes
        attempts.append({**hr_params, "detections": len(retry_boxes)})
        retry_used = True

    # 4. Tiled Inference Pass
    tiled_boxes, tiled_metadata = run_tiled_inference(
        image, model, frame_width, frame_height, 
        min_rel_area=min_rel_area, max_rel_area=max_rel_area,
        min_aspect=min_aspect, max_aspect=max_aspect,
        geo_filter_enabled=geo_filter_enabled,
        tile_size=tile_size, overlap_ratio=overlap_ratio,
        tile_imgsz=tile_imgsz, verbose=verbose
    )

    # Combine all
    all_boxes = person_boxes + clahe_boxes + retry_boxes + tiled_boxes
    before_nms_total = len(all_boxes)
    raw_detections_before_geo_filter = (
        after_class_filter_ff
        + after_class_filter_clahe
        + after_class_filter_retry
        + tiled_metadata["after_class_filter_tiled"]
    )
    detections_after_geo_filter = (
        after_geometry_filter_ff
        + len(clahe_boxes)
        + len(retry_boxes)
        + tiled_metadata["after_geometry_filter_tiled"]
    )
    
    # Global NMS
    final_boxes = global_nms(all_boxes, nms_iou)
    final_detections = len(final_boxes)

    metadata = {
        "mode": "drone_demo",
        "model": model_name_actual,
        "model_path": model_name_actual,
        "preset": preset,
        "imgsz": imgsz,
        "conf": conf,
        "classes": classes,
        "geo_filter_enabled": geo_filter_enabled,
        "retry_used": retry_used,
        "attempts": attempts,
        "full_frame_attempt": True,
        "final_detection_stage": "hybrid",
        "full_frame_raw": full_frame_raw,
        "clahe_raw": clahe_raw,
        "high_res_retry_raw": high_res_retry_raw,
        "tiled_raw_total": tiled_metadata["tiled_raw_total"],
        "after_class_filter": raw_detections_before_geo_filter,
        "raw_detections_before_geo_filter": raw_detections_before_geo_filter,
        "after_geometry_filter": detections_after_geo_filter,
        "detections_after_geo_filter": detections_after_geo_filter,
        "before_nms_total": before_nms_total,
        "after_nms": final_detections,
        "final_detections": final_detections,
        "nms_iou": nms_iou,
        "tile_size": tile_size,
        "tile_imgsz": tile_imgsz,
        "overlap_ratio": overlap_ratio,
        "min_rel_area": min_rel_area,
        "max_rel_area": max_rel_area,
        "min_aspect": min_aspect,
        "max_aspect": max_aspect,
        **tiled_metadata,
    }

    payload = build_detection_payload(image, image_path, final_boxes, frame_width, frame_height, metadata)

    json_path = out_dir / "detections.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        
    print("[YOLO_SCHEMA] detections.json policy schema active")
    m = payload.get("metadata", {})
    print(f"[YOLO_SCHEMA] confirmed={m.get('confirmed_count')} candidate={m.get('candidate_count')} rejected={m.get('rejected_count')} review_required={m.get('review_required')}")

    # Defensive check
    for d in payload.get("detections", []):
        if "status" not in d:
            print("[ERROR] Detection payload missing status fields")
            break

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
    parser.add_argument("--preset", choices=["fast", "high_recall"], default="fast", help="Inference preset")
    parser.add_argument("--model", default=None, help="YOLO model path/name. Default prefers models/drone_tactical_best.pt, then yolov8s.pt, then yolov8n.pt.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_DRONE_IMGSZ, help=f"Inference image size (default: {DEFAULT_DRONE_IMGSZ})")
    parser.add_argument("--conf", type=float, default=DEFAULT_DRONE_CONF, help=f"Confidence threshold (default: {DEFAULT_DRONE_CONF})")
    parser.add_argument("--classes", default=None, help="Comma-separated YOLO class IDs, 'all', or empty. Kept for metadata only; person filtering is name-based.")
    parser.add_argument("--min-rel-area", type=float, default=None, help="Minimum relative area for geometric filter. Preset default: fast=0.00005, high_recall=0.00002")
    parser.add_argument("--max-rel-area", type=float, default=None, help="Maximum relative area for geometric filter. Preset default: fast=0.07, high_recall=0.08")
    parser.add_argument("--min-aspect", type=float, default=None, help="Minimum height/width aspect ratio for geometric filter. Preset default: 0.25")
    parser.add_argument("--max-aspect", type=float, default=None, help="Maximum height/width aspect ratio for geometric filter. Preset default: 3.0")
    parser.add_argument("--no-geo-filter", action="store_true", help="Disable geometric filtering after person-name filtering.")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for tiled inference (default: 640)")
    parser.add_argument("--overlap-ratio", type=float, default=0.35, help="Overlap ratio for tiled inference (default: 0.35)")
    parser.add_argument("--tile-imgsz", type=int, default=960, help="Inference resolution per tile (default: 960)")
    parser.add_argument("--nms-iou", type=float, default=0.65, help="NMS IoU threshold (default: 0.65)")
    parser.add_argument("--no-retry", action="store_true", help="Disable empty-result retry with larger image size and lower confidence.")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    
    # Apply presets
    if args.preset == "high_recall":
        args.tile_size = 480
        args.tile_imgsz = 1280
        args.overlap_ratio = 0.35
        args.nms_iou = 0.65
        if args.min_rel_area is None:
            args.min_rel_area = 0.00002
        if args.max_rel_area is None:
            args.max_rel_area = 0.08
        if args.min_aspect is None:
            args.min_aspect = 0.25
        if args.max_aspect is None:
            args.max_aspect = 3.0
    else:
        # fast
        args.tile_size = 640
        args.tile_imgsz = 960
        args.overlap_ratio = 0.35
        args.nms_iou = 0.65
        if args.min_rel_area is None:
            args.min_rel_area = 0.00005
        if args.max_rel_area is None:
            args.max_rel_area = 0.07
        if args.min_aspect is None:
            args.min_aspect = 0.25
        if args.max_aspect is None:
            args.max_aspect = 3.0

    try:
        run_yolo_detection(
            args.image,
            model_name=args.model,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=parse_classes(args.classes),
            retry_if_empty=not args.no_retry,
            min_rel_area=args.min_rel_area,
            max_rel_area=args.max_rel_area,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            geo_filter_enabled=not args.no_geo_filter,
            preset=args.preset,
            tile_size=args.tile_size,
            overlap_ratio=args.overlap_ratio,
            tile_imgsz=args.tile_imgsz,
            nms_iou=args.nms_iou,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] {exc}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
