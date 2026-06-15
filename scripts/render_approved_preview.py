from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import pathlib
from typing import Any

import cv2


DEFAULT_OUTPUT = pathlib.Path("frontend/public/data/approved_detection_preview.jpg")
DEFAULT_JSON_OUTPUT = pathlib.Path("frontend/public/data/approved_detections.json")
BOX_COLOR = (22, 163, 74)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render clean preview boxes from manually approved YOLO labels."
    )
    parser.add_argument("--image", required=True, type=pathlib.Path, help="Approved image path.")
    parser.add_argument(
        "--labels",
        type=pathlib.Path,
        default=None,
        help="YOLO label path. Defaults to matching path under labels/ with .txt suffix.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"Output preview image path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--class-name", default="person", help="Class display name (default: person).")
    return parser.parse_args()


def infer_label_path(image_path: pathlib.Path) -> pathlib.Path:
    parts = list(image_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return pathlib.Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_to_pixels(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1 = int(round((cx - bw / 2) * image_width))
    y1 = int(round((cy - bh / 2) * image_height))
    x2 = int(round((cx + bw / 2) * image_width))
    y2 = int(round((cy + bh / 2) * image_height))
    x1 = int(clamp(x1, 0, image_width - 1))
    y1 = int(clamp(y1, 0, image_height - 1))
    x2 = int(clamp(x2, 0, image_width - 1))
    y2 = int(clamp(y2, 0, image_height - 1))
    return x1, y1, x2, y2


def read_approved_labels(label_path: pathlib.Path, image_width: int, image_height: int) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
        return detections

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            print(f"[WARN] Skipping malformed label line {label_path}:{line_number}: {line}")
            continue
        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = [float(value) for value in parts[1:]]
        except ValueError:
            print(f"[WARN] Skipping non-numeric label line {label_path}:{line_number}: {line}")
            continue

        if any(value < 0.0 or value > 1.0 for value in (cx, cy, bw, bh)):
            print(f"[WARN] Skipping out-of-range label line {label_path}:{line_number}: {line}")
            continue

        x1, y1, x2, y2 = normalized_to_pixels(cx, cy, bw, bh, image_width, image_height)
        detections.append({
            "class_id": class_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "xyxy": [x1, y1, x2, y2],
            "normalized_bbox": {
                "x_center": cx,
                "y_center": cy,
                "width": bw,
                "height": bh,
            },
        })

    return detections


def draw_detections(image, detections: list[dict[str, Any]], class_name: str) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"approved {class_name}"
        label_y = max(18, y1 - 8)
        cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2, cv2.LINE_AA)


def write_json(
    image_path: pathlib.Path,
    label_path: pathlib.Path,
    output_path: pathlib.Path,
    detections: list[dict[str, Any]],
) -> pathlib.Path:
    DEFAULT_JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "manual_approved_labels",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "image_path": str(image_path),
        "label_path": str(label_path),
        "output_preview_path": str(output_path),
        "detection_count": len(detections),
        "detections": detections,
    }
    DEFAULT_JSON_OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return DEFAULT_JSON_OUTPUT


def main() -> int:
    args = parse_args()
    image_path = args.image
    label_path = args.labels or infer_label_path(image_path)
    output_path = args.output

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]
    detections = read_approved_labels(label_path, image_width, image_height)
    draw_detections(image, detections, args.class_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    json_path = write_json(image_path, label_path, output_path, detections)

    print(f"[OK] Rendered {len(detections)} approved detections -> {output_path}")
    print(f"[OK] Wrote metadata -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
