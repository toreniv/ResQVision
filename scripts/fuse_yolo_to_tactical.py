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


def extract_detection_payload(raw: Any) -> tuple[list[dict[str, Any]], int, int]:
    frame_width = DEFAULT_FRAME_WIDTH
    frame_height = DEFAULT_FRAME_HEIGHT

    if isinstance(raw, dict):
        frame_width = int(as_float(raw.get("frame_width") or raw.get("width"), DEFAULT_FRAME_WIDTH))
        frame_height = int(as_float(raw.get("frame_height") or raw.get("height"), DEFAULT_FRAME_HEIGHT))
        detections = raw.get("detections") or raw.get("targets") or []
    elif isinstance(raw, list):
        detections = raw
    else:
        detections = []

    return [d for d in detections if isinstance(d, dict)], frame_width, frame_height


def bbox_center(det: dict[str, Any]) -> tuple[float, float] | None:
    bbox = det.get("bbox") or det.get("box")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None

    x = as_float(bbox[0])
    y = as_float(bbox[1])
    third = as_float(bbox[2])
    fourth = as_float(bbox[3])
    fmt = str(det.get("bbox_format") or det.get("format") or "xywh").lower()

    if fmt in {"xyxy", "x1y1x2y2"}:
        return (x + third) / 2, (y + fourth) / 2

    return x + third / 2, y + fourth / 2


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
    return str(det.get("class") or det.get("label") or det.get("name") or "").lower()


def main() -> int:
    raw_detections = load_json(DETECTIONS_PATH, None)
    detections, frame_width, frame_height = extract_detection_payload(raw_detections)

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
        center = bbox_center(det)
        if center is None:
            continue

        center_x, center_y = center
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
            "class": det.get("class") or det.get("label") or "person",
            "confidence": confidence,
            "bbox": det.get("bbox") or det.get("box"),
            "x_map": (center_x / max(frame_width, 1)) * MAP_SIZE,
            "y_map": (center_y / max(frame_height, 1)) * MAP_SIZE,
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

    write_fusion("YOLO_LIVE", targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
