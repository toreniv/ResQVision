from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import pathlib
from typing import Any

import cv2


DEFAULT_OUTPUT_IMAGE = pathlib.Path("frontend/public/data/human_review_preview.jpg")
DEFAULT_OUTPUT_JSON = pathlib.Path("frontend/public/data/human_review_detections.json")
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
MIN_BOX_SIZE = 3
BOX_COLOR = (22, 163, 74)
PREVIEW_COLOR = (59, 130, 246)
SELECTED_COLOR = (0, 255, 255)
WINDOW_NAME = "ResQVision Human-Reviewed Demo Annotation"

Box = tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a human-reviewed demo preview without modifying training data."
    )
    parser.add_argument("--image", required=True, type=pathlib.Path, help="Image to annotate for demo output.")
    parser.add_argument(
        "--output-image",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_IMAGE,
        help=f"Output preview image path (default: {DEFAULT_OUTPUT_IMAGE}).",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument("--class-name", default="person", help="Class display name (default: person).")
    parser.add_argument(
        "--label-mode",
        choices=["none", "first", "all"],
        default="none",
        help="Preview label rendering mode: none, first, or all (default: none).",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def display_scale(width: int, height: int) -> float:
    return min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)


def image_to_display(point: tuple[float, float], scale: float) -> tuple[int, int]:
    x, y = point
    return int(round(x * scale)), int(round(y * scale))


def display_to_image(point: tuple[int, int], scale: float, width: int, height: int) -> tuple[float, float]:
    x, y = point
    return clamp(x / scale, 0, width - 1), clamp(y / scale, 0, height - 1)


def ordered_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    left = clamp(min(x1, x2), 0, width - 1)
    right = clamp(max(x1, x2), 0, width - 1)
    top = clamp(min(y1, y2), 0, height - 1)
    bottom = clamp(max(y1, y2), 0, height - 1)
    return left, top, right, bottom


def nearest_box(boxes: list[Box], point: tuple[float, float]) -> int | None:
    if not boxes:
        return None
    px, py = point
    best_index = None
    best_distance = float("inf")
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if left <= px <= right and top <= py <= bottom:
            distance = 0.0
        else:
            cx = (left + right) / 2
            cy = (top + bottom) / 2
            distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def box_to_detection(box: Box, width: int, height: int, class_name: str) -> dict[str, Any]:
    x1, y1, x2, y2 = ordered_box(box, width, height)
    box_width = x2 - x1
    box_height = y2 - y1
    return {
        "class": class_name,
        "bbox": [int(round(x1)), int(round(y1)), int(round(box_width)), int(round(box_height))],
        "xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
        "normalized_bbox": {
            "x_center": round(((x1 + x2) / 2) / width, 6),
            "y_center": round(((y1 + y2) / 2) / height, 6),
            "width": round(box_width / width, 6),
            "height": round(box_height / height, 6),
        },
    }


def draw_label(image, text: str, x: int, y: int, scale: float = 0.5) -> None:
    label_y = max(16, y - 7)
    cv2.putText(image, text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, scale, BOX_COLOR, 1, cv2.LINE_AA)


def draw_boxes(
    image,
    boxes: list[Box],
    width: int,
    height: int,
    label: str,
    selected_index: int | None = None,
    label_mode: str = "none",
) -> None:
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = ordered_box(box, width, height)
        color = SELECTED_COLOR if index == selected_index else BOX_COLOR
        thickness = 3 if index == selected_index else 2
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(image, p1, p2, color, thickness)
        should_label = label_mode == "all" or (label_mode == "first" and index == 0)
        if should_label:
            draw_label(image, label, p1[0], p1[1])


def draw_instruction_overlay(image, box_count: int) -> None:
    lines = [
        "Mouse drag: draw | Click: select | d: delete | z: undo | c: clear",
        "p: export preview + JSON | q: quit without exporting",
        f"Boxes: {box_count}",
    ]
    y = 24
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def detections_to_boxes(json_path: pathlib.Path, image_path: pathlib.Path) -> list[Box]:
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    saved_image = pathlib.Path(data.get("image_path", ""))
    if saved_image.name and saved_image.name != image_path.name:
        return []
    boxes = []
    for detection in data.get("detections", []):
        xyxy = detection.get("xyxy")
        if isinstance(xyxy, list) and len(xyxy) == 4:
            boxes.append(tuple(float(value) for value in xyxy))
    return boxes


class DemoAnnotator:
    def __init__(self, image, initial_boxes: list[Box] | None = None) -> None:
        self.image = image
        self.height, self.width = image.shape[:2]
        self.scale = display_scale(self.width, self.height)
        self.boxes: list[Box] = initial_boxes[:] if initial_boxes else []
        self.history: list[int] = []
        self.selected_index: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_current: tuple[float, float] | None = None

    def mouse_callback(self, event, x, y, _flags, _param) -> None:
        image_point = display_to_image((x, y), self.scale, self.width, self.height)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = image_point
            self.drag_current = image_point
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = image_point
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            start = self.drag_start
            end = image_point
            self.drag_start = None
            self.drag_current = None
            if abs(end[0] - start[0]) < MIN_BOX_SIZE or abs(end[1] - start[1]) < MIN_BOX_SIZE:
                self.selected_index = nearest_box(self.boxes, image_point)
                return
            self.boxes.append((start[0], start[1], end[0], end[1]))
            self.selected_index = len(self.boxes) - 1
            self.history.append(self.selected_index)

    def delete_selected(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.boxes):
            return
        deleted = self.selected_index
        del self.boxes[deleted]
        self.history = [index for index in self.history if index != deleted]
        self.history = [index - 1 if index > deleted else index for index in self.history]
        self.selected_index = None

    def undo_last(self) -> None:
        while self.history:
            index = self.history.pop()
            if 0 <= index < len(self.boxes):
                del self.boxes[index]
                self.history = [i - 1 if i > index else i for i in self.history]
                self.selected_index = None
                return

    def clear_all(self) -> None:
        self.boxes.clear()
        self.history.clear()
        self.selected_index = None

    def render(self):
        display = cv2.resize(self.image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        scaled_boxes = [
            (box[0] * self.scale, box[1] * self.scale, box[2] * self.scale, box[3] * self.scale)
            for box in self.boxes
        ]
        draw_boxes(
            display,
            scaled_boxes,
            display.shape[1],
            display.shape[0],
            "human-reviewed person",
            self.selected_index,
            label_mode="first",
        )
        if self.drag_start is not None and self.drag_current is not None:
            x1, y1 = image_to_display(self.drag_start, self.scale)
            x2, y2 = image_to_display(self.drag_current, self.scale)
            cv2.rectangle(display, (x1, y1), (x2, y2), PREVIEW_COLOR, 2)
        draw_instruction_overlay(display, len(self.boxes))
        return display


def export_demo(
    image,
    boxes: list[Box],
    image_path: pathlib.Path,
    output_image: pathlib.Path,
    output_json: pathlib.Path,
    class_name: str,
    label_mode: str,
) -> None:
    height, width = image.shape[:2]
    rendered = image.copy()
    draw_boxes(rendered, boxes, width, height, f"human-reviewed {class_name}", label_mode=label_mode)

    detections = [
        box_to_detection(box, width, height, class_name)
        for box in boxes
        if abs(box[2] - box[0]) >= MIN_BOX_SIZE and abs(box[3] - box[1]) >= MIN_BOX_SIZE
    ]

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), rendered)

    payload = {
        "source": "human_review_demo",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "image_path": str(image_path),
        "output_image_path": str(output_image),
        "detection_count": len(detections),
        "detections": detections,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_annotation(
    image,
    image_path: pathlib.Path,
    output_image: pathlib.Path,
    output_json: pathlib.Path,
    class_name: str,
    label_mode: str,
) -> bool:
    initial_boxes = detections_to_boxes(output_json, image_path)
    annotator = DemoAnnotator(image, initial_boxes)
    if initial_boxes:
        print(f"[DEMO] Loaded {len(initial_boxes)} existing demo boxes from {output_json}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(annotator.width * annotator.scale), int(annotator.height * annotator.scale))
    cv2.setMouseCallback(WINDOW_NAME, annotator.mouse_callback)

    while True:
        cv2.imshow(WINDOW_NAME, annotator.render())
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key == ord("q"):
            cv2.destroyWindow(WINDOW_NAME)
            return False
        if key == ord("p"):
            export_demo(image, annotator.boxes, image_path, output_image, output_json, class_name, label_mode)
            print(f"[DEMO] Boxes: {len(annotator.boxes)}")
            print(f"[OK] Exported preview -> {output_image}")
            print(f"[OK] Exported JSON -> {output_json}")
            cv2.destroyWindow(WINDOW_NAME)
            return True
        if key == ord("d"):
            annotator.delete_selected()
        elif key == ord("z"):
            annotator.undo_last()
        elif key == ord("c"):
            annotator.clear_all()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    print(f"[DEMO] Loaded image: {args.image}")
    exported = run_annotation(image, args.image, args.output_image, args.output_json, args.class_name, args.label_mode)
    if not exported:
        print("[DEMO] Quit without exporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
