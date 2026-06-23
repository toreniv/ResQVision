from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
DETECTIONS_PATH = DATA_DIR / "detections.json"
MANUAL_MARKERS_PATH = DATA_DIR / "manual_markers.json"
RISK_RANKING_PATH = DATA_DIR / "risk_ranking.json"
FUSION_PATH = DATA_DIR / "tactical_fusion.json"

DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
MAP_SIZE = 1000
LOCALIZATION_MODE = "visual_relative"
LOCALIZATION_LABEL = "GPS-Denied Visual Fix"
MANUAL_LOCALIZATION_MODE = "manual_visual_relative"
MANUAL_LOCALIZATION_LABEL = "Manual Drone Visual Fix"


def load_json(path: Path, fallback: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback


def write_fusion(fusion_mode: str, targets: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"fusion_mode": fusion_mode, "fusion_status": fusion_mode, "targets": targets}
    if metadata:
        payload.update(metadata)
    with FUSION_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_detection_payload(raw: Any) -> tuple[list[dict[str, Any]], int, int, str]:
    if isinstance(raw, list):
        print("[WARN] Legacy detections array detected. Treating it as unified YOLO schema internally.")
        raw = {
            "source": "legacy_array",
            "frame_width": DEFAULT_FRAME_WIDTH,
            "frame_height": DEFAULT_FRAME_HEIGHT,
            "detections": raw,
        }

    if not isinstance(raw, dict):
        return [], DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, ""

    source = str(raw.get("source") or "")
    detections = raw.get("detections", [])
    frame_width = int(as_float(raw.get("frame_width"), DEFAULT_FRAME_WIDTH))
    frame_height = int(as_float(raw.get("frame_height"), DEFAULT_FRAME_HEIGHT))

    if not source or "frame_width" not in raw or "frame_height" not in raw or "detections" not in raw:
        print("[WARN] detections.json is missing unified schema fields. Expected source, frame_width, frame_height, detections.")

    if not isinstance(detections, list):
        detections = []

    return [d for d in detections if isinstance(d, dict)], frame_width, frame_height, source


def fusion_mode_from_source(source: str) -> str:
    if source == "browser_transformers":
        return "BROWSER_TRANSFORMERS"
    if source == "live_camera":
        return "YOLO_LIVE"
    if source == "offline_image":
        return "YOLO_IMAGE"
    return "YOLO_FUSION"


def detection_center(det: dict[str, Any]) -> tuple[float, float] | None:
    center = det.get("center")
    if isinstance(center, list) and len(center) >= 2:
        return as_float(center[0]), as_float(center[1])

    bbox = det.get("bbox") or det.get("bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox_xyxy = det.get("bbox_xyxy")
        if isinstance(bbox_xyxy, list) and len(bbox_xyxy) >= 4:
            x1 = as_float(bbox_xyxy[0])
            y1 = as_float(bbox_xyxy[1])
            x2 = as_float(bbox_xyxy[2])
            y2 = as_float(bbox_xyxy[3])
            return (x1 + x2) / 2, (y1 + y2) / 2
        return None

    print(f"[WARN] Detection {det.get('id', '?')} is missing center. Falling back to bbox center.")

    x = as_float(bbox[0])
    y = as_float(bbox[1])
    width = as_float(bbox[2])
    height = as_float(bbox[3])

    return x + width / 2, y + height / 2


def visual_localization(center_x: float, center_y: float, frame_width: int, frame_height: int) -> dict[str, Any]:
    x_map = round((center_x / max(frame_width, 1)) * MAP_SIZE, 1)
    y_map = round((center_y / max(frame_height, 1)) * MAP_SIZE, 1)
    image_center = [round(center_x, 1), round(center_y, 1)]
    map_position = [x_map, y_map]
    return {
        "localization_mode": LOCALIZATION_MODE,
        "localization_label": LOCALIZATION_LABEL,
        "image_center": image_center,
        "map_position": map_position,
        "x_map": x_map,
        "y_map": y_map,
    }


def risk_from_entry(entry: Any) -> float | None:
    if not isinstance(entry, dict):
        return None
    raw = entry.get("risk_score", entry.get("risk", entry.get("risk_percent")))
    score = as_float(raw, -1)
    if score < 0:
        return None
    return score / 100 if score > 1 else score


def category_from_risk(risk_score: float) -> str:
    if risk_score >= 0.8:
        return "critical"
    if risk_score >= 0.6:
        return "urgent"
    return "stable"


def recommended_action(risk_score: float) -> str:
    if risk_score >= 0.8:
        return "Immediate evacuation"
    if risk_score >= 0.6:
        return "Monitor closely"
    return "Low priority"


def detection_class(det: dict[str, Any]) -> str:
    return str(det.get("class") or det.get("class_name") or "").lower()


def risk_id(entry: dict[str, Any]) -> str:
    return str(entry.get("soldier_id") or entry.get("id") or entry.get("SoldierID") or "")


def risk_lookup(entries: list[Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = risk_id(entry)
        if entry_id:
            lookup[entry_id] = entry
    return lookup


def category_from_status(status: Any) -> str:
    value = str(status or "").lower()
    if value in {"critical", "urgent", "stable"}:
        return value
    return "stable"


def marker_float(marker: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in marker:
            return as_float(marker.get(key), default)
    return default


def manual_marker_target(marker: dict[str, Any], index: int, lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    soldier_id = str(marker.get("soldier_id") or marker.get("id") or f"manual_{index + 1}")
    x_norm = marker_float(marker, "x_norm", default=-1)
    y_norm = marker_float(marker, "y_norm", default=-1)
    x_map = marker_float(marker, "x_map", "map_x", default=x_norm * MAP_SIZE if x_norm >= 0 else 0)
    y_map = marker_float(marker, "y_map", "map_y", default=y_norm * MAP_SIZE if y_norm >= 0 else 0)
    x_image = marker_float(marker, "x_image", default=0)
    y_image = marker_float(marker, "y_image", default=0)

    matched = lookup.get(soldier_id)
    matched_risk = risk_from_entry(matched) if matched else None
    telemetry_status = "LINKED" if matched and matched_risk is not None else "UNLINKED"
    category = (
        matched.get("category") if matched else None
    ) or category_from_status(marker.get("status"))

    target: dict[str, Any] = {
        "id": soldier_id,
        "soldier_id": soldier_id,
        "band_id": marker.get("band_id") or "",
        "source": "MANUAL_TAG",
        "class": "visual_point",
        "confidence": 1.0,
        "label": marker.get("label") or "Soldier",
        "status": marker.get("status") or "",
        "x_image": round(x_image, 1),
        "y_image": round(y_image, 1),
        "x_norm": round(x_map / MAP_SIZE, 4),
        "y_norm": round(y_map / MAP_SIZE, 4),
        "image_center": [round(x_image, 1), round(y_image, 1)],
        "x_map": round(x_map, 1),
        "y_map": round(y_map, 1),
        "map_x": round(x_map, 1),
        "map_y": round(y_map, 1),
        "map_position": [round(x_map, 1), round(y_map, 1)],
        "localization_mode": MANUAL_LOCALIZATION_MODE,
        "localization_label": MANUAL_LOCALIZATION_LABEL,
        "telemetry_status": telemetry_status,
        "risk_score": matched_risk,
        "category": category,
        "hr": matched.get("heart_rate") or matched.get("hr") or 0 if matched else 0,
        "spo2": matched.get("spo2") or matched.get("SpO2") or 0 if matched else 0,
        "recommended_action": recommended_action(matched_risk) if matched_risk is not None else "Assign ResQBand / Soldier ID",
        "priority": None,
        "rank": None,
    }
    return target


def main() -> int:
    raw_detections = load_json(DETECTIONS_PATH, None)
    detections, frame_width, frame_height, detection_source = extract_detection_payload(raw_detections)

    risk_ranking = load_json(RISK_RANKING_PATH, [])
    if isinstance(risk_ranking, list):
        risk_entries = risk_ranking
    elif isinstance(risk_ranking, dict):
        risk_entries = risk_ranking.get("targets", [])
    else:
        risk_entries = []
    risk_by_id = risk_lookup(risk_entries)
    person_detections = []
    yolo_confirmed_count = 0
    yolo_candidate_count = 0
    yolo_rejected_count = 0
    source_label = "browser_transformers" if detection_source == "browser_transformers" else "YOLO"

    for det in detections:
        if detection_class(det) not in {"person", "soldier", "casualty", "0", ""}:
            continue
        status = det.get("status", "confirmed")
        if status == "confirmed":
            person_detections.append(det)
            yolo_confirmed_count += 1
        elif status == "candidate":
            yolo_candidate_count += 1
        elif status.startswith("rejected"):
            yolo_rejected_count += 1

    targets: list[dict[str, Any]] = []
    for index, det in enumerate(person_detections):
        center = detection_center(det)
        if center is None:
            continue

        center_x, center_y = center
        localization = visual_localization(center_x, center_y, frame_width, frame_height)
        confidence = as_float(det.get("confidence", det.get("conf")), 0.0)
        matched = risk_entries[index] if index < len(risk_entries) and isinstance(risk_entries[index], dict) else {}
        matched_risk = risk_from_entry(matched)
        risk_score = matched_risk if matched_risk is not None else min(0.95, confidence * 1.05)
        risk_score = max(0.0, min(1.0, risk_score))
        target_id = matched.get("soldier_id") or matched.get("id") or det.get("id") or f"YOLO-{index + 1:03d}"

        targets.append({
            "id": str(target_id),
            "source": "BROWSER_TRANSFORMERS" if detection_source == "browser_transformers" else "YOLO",
            "detection_index": index,
            "class": det.get("class") or det.get("class_name") or "person",
            "confidence": confidence,
            "bbox": det.get("bbox") or det.get("bbox_xywh"),
            "image_center": localization["image_center"],
            "x_map": localization["x_map"],
            "y_map": localization["y_map"],
            "map_position": localization["map_position"],
            "localization_mode": localization["localization_mode"],
            "localization_label": localization["localization_label"],
            "risk_score": risk_score,
            "category": matched.get("category") or category_from_risk(risk_score),
            "hr": matched.get("heart_rate") or matched.get("hr") or 0,
            "spo2": matched.get("spo2") or matched.get("SpO2") or 0,
            "recommended_action": recommended_action(risk_score),
        })

    manual_markers = load_json(MANUAL_MARKERS_PATH, [])
    if not isinstance(manual_markers, list):
        manual_markers = []

    manual_targets = [
        manual_marker_target(marker, index, risk_by_id)
        for index, marker in enumerate(manual_markers)
        if isinstance(marker, dict)
    ]

    if not targets and not manual_targets:
        metadata = {
            "confirmed_count": yolo_confirmed_count,
            "candidate_count": yolo_candidate_count,
            "rejected_count": yolo_rejected_count,
            "yolo_confirmed_count": yolo_confirmed_count,
            "yolo_candidate_count": yolo_candidate_count,
            "yolo_rejected_count": yolo_rejected_count,
            "source": source_label,
            "confidence_policy": "confirmed_only_for_tactical_fusion",
        }
        if yolo_candidate_count > 0:
            metadata["review_required"] = True
            write_fusion("YOLO_CANDIDATES_REQUIRE_REVIEW", [], metadata)
        else:
            write_fusion("NO_CONFIRMED_YOLO_TARGETS", [], metadata)
        return 0

    targets.sort(key=lambda target: target["risk_score"], reverse=True)
    for priority, target in enumerate(targets, start=1):
        target["priority"] = priority
        target["rank"] = priority

    combined_targets = targets + manual_targets
    fusion_mode = fusion_mode_from_source(detection_source) if targets else "MANUAL_TAGS"
    if targets and manual_targets:
        fusion_mode = f"{fusion_mode}_MANUAL"

    metadata = {
        "confirmed_count": yolo_confirmed_count,
        "candidate_count": yolo_candidate_count,
        "rejected_count": yolo_rejected_count,
        "yolo_confirmed_count": yolo_confirmed_count,
        "yolo_candidate_count": yolo_candidate_count,
        "yolo_rejected_count": yolo_rejected_count,
        "source": source_label,
        "confidence_policy": "confirmed_only_for_tactical_fusion",
        "review_required": False,
    }

    write_fusion(fusion_mode, combined_targets, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
