from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
DETECTIONS_PATH = DATA_DIR / "detections.json"
RISK_RANKING_PATH = DATA_DIR / "risk_ranking.json"
FUSION_PATH = DATA_DIR / "tactical_fusion.json"

DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
MAP_SIZE = 1000
LOCALIZATION_MODE = "visual_relative"
LOCALIZATION_LABEL = "GPS-Denied Visual Fix"


def load_json(path: Path, fallback: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback


def write_fusion(fusion_mode: str, targets: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FUSION_PATH.open("w", encoding="utf-8") as handle:
        json.dump({"fusion_mode": fusion_mode, "targets": targets}, handle, indent=2)
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
    if source == "live_camera":
        return "YOLO_LIVE"
    if source == "offline_image":
        return "YOLO_IMAGE"
    return "YOLO_FUSION"


def detection_center(det: dict[str, Any]) -> tuple[float, float] | None:
    center = det.get("center")
    if isinstance(center, list) and len(center) >= 2:
        return as_float(center[0]), as_float(center[1])

    bbox = det.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
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
    return str(det.get("class") or "").lower()


def main() -> int:
    raw_detections = load_json(DETECTIONS_PATH, None)
    detections, frame_width, frame_height, detection_source = extract_detection_payload(raw_detections)

    if not detections:
        write_fusion("NO_DATA", [])
        return 0

    risk_ranking = load_json(RISK_RANKING_PATH, [])
    if isinstance(risk_ranking, list):
        risk_entries = risk_ranking
    elif isinstance(risk_ranking, dict):
        risk_entries = risk_ranking.get("targets", [])
    else:
        risk_entries = []
    person_detections = [
        det for det in detections
        if detection_class(det) in {"person", "soldier", "casualty", "0", ""}
    ]

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
            "source": "YOLO",
            "detection_index": index,
            "class": det.get("class") or "person",
            "confidence": confidence,
            "bbox": det.get("bbox"),
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

    if not targets:
        write_fusion("NO_DATA", [])
        return 0

    targets.sort(key=lambda target: target["risk_score"], reverse=True)
    for priority, target in enumerate(targets, start=1):
        target["priority"] = priority
        target["rank"] = priority

    write_fusion(fusion_mode_from_source(detection_source), targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
