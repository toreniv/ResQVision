from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import cv2


DRAFT_DIR = pathlib.Path("dataset_draft")
DRAFT_IMAGES_DIR = DRAFT_DIR / "images"
DRAFT_LABELS_DIR = DRAFT_DIR / "labels"
MANIFEST_PATH = DRAFT_DIR / "manifest.json"
APPROVED_DATASET_ROOT = pathlib.Path("dataset_approved_v2")
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
MIN_BOX_SIZE = 3
WINDOW_NAME = "ResQVision Dataset Review"

Box = tuple[float, float, float, float]


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually review and edit draft YOLO labels.")
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=APPROVED_DATASET_ROOT,
        help="Approved dataset root. Defaults to dataset_approved_v2.",
    )
    parser.add_argument(
        "--review-status",
        action="append",
        default=None,
        choices=["unreviewed", "skipped", "approved", "background"],
        help="Manifest status to include. Defaults to unreviewed and skipped.",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_box(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = clamp(min(x1, x2), 0, width - 1)
    x2 = clamp(max(x1, x2), 0, width - 1)
    y1 = clamp(min(y1, y2), 0, height - 1)
    y2 = clamp(max(y1, y2), 0, height - 1)
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return (
        clamp(cx, 0.0, 1.0),
        clamp(cy, 0.0, 1.0),
        clamp(bw, 0.0, 1.0),
        clamp(bh, 0.0, 1.0),
    )


def load_yolo_boxes(label_path: pathlib.Path, width: int, height: int) -> list[Box]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id, cx, cy, bw, bh = parts
            if class_id != "0":
                continue
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        except ValueError:
            continue
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        if abs(x2 - x1) >= MIN_BOX_SIZE and abs(y2 - y1) >= MIN_BOX_SIZE:
            boxes.append((x1, y1, x2, y2))
    return boxes


def write_yolo_labels(label_path: pathlib.Path, boxes: list[Box], width: int, height: int) -> None:
    lines = []
    for box in boxes:
        cx, cy, bw, bh = normalize_box(box, width, height)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def copy_approved_image(image_path: pathlib.Path, approved_images_dir: pathlib.Path) -> None:
    approved_images_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, approved_images_dir / image_path.name)


def approve_boxes(
    image_path: pathlib.Path,
    label_path: pathlib.Path,
    approved_images_dir: pathlib.Path,
    approved_labels_dir: pathlib.Path,
    boxes: list[Box],
    width: int,
    height: int,
) -> None:
    copy_approved_image(image_path, approved_images_dir)
    approved_labels_dir.mkdir(parents=True, exist_ok=True)
    write_yolo_labels(approved_labels_dir / label_path.name, boxes, width, height)


def approve_background(
    image_path: pathlib.Path,
    label_path: pathlib.Path,
    approved_images_dir: pathlib.Path,
    approved_labels_dir: pathlib.Path,
) -> None:
    copy_approved_image(image_path, approved_images_dir)
    approved_labels_dir.mkdir(parents=True, exist_ok=True)
    (approved_labels_dir / label_path.name).write_text("", encoding="utf-8")


def count_status(manifest: dict, status: str) -> int:
    return sum(1 for row in manifest.values() if row.get("status") == status)


def display_scale(width: int, height: int) -> float:
    return min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)


def image_to_display(point: tuple[float, float], scale: float) -> tuple[int, int]:
    x, y = point
    return int(round(x * scale)), int(round(y * scale))


def display_to_image(point: tuple[int, int], scale: float, width: int, height: int) -> tuple[float, float]:
    x, y = point
    return clamp(x / scale, 0, width - 1), clamp(y / scale, 0, height - 1)


def nearest_box(boxes: list[Box], point: tuple[float, float]) -> int | None:
    if not boxes:
        return None
    px, py = point
    best_index = None
    best_distance = float("inf")
    for index, (x1, y1, x2, y2) in enumerate(boxes):
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


def draw_overlay(
    display_image,
    boxes: list[Box],
    scale: float,
    selected_index: int | None,
    draft_count: int,
    drag_box: Box | None,
) -> None:
    for index, box in enumerate(boxes):
        x1, y1 = image_to_display((box[0], box[1]), scale)
        x2, y2 = image_to_display((box[2], box[3]), scale)
        color = (0, 255, 255) if index == selected_index else (22, 163, 74)
        thickness = 3 if index == selected_index else 2
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)

    if drag_box is not None:
        x1, y1 = image_to_display((drag_box[0], drag_box[1]), scale)
        x2, y2 = image_to_display((drag_box[2], drag_box[3]), scale)
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (59, 130, 246), 2)

    lines = [
        "Draw box: mouse drag | Approve: a | Background: e | Skip: s",
        "Delete selected: d | Undo last: z | Clear all: c | Quit: q",
        f"Boxes: {len(boxes)} | Draft boxes loaded: {draft_count}",
    ]
    y = 24
    for line in lines:
        cv2.putText(display_image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display_image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


class BoxEditor:
    def __init__(self, image, boxes: list[Box]) -> None:
        self.image = image
        self.height, self.width = image.shape[:2]
        self.scale = display_scale(self.width, self.height)
        self.boxes = boxes[:]
        self.draft_count = len(boxes)
        self.manual_history: list[int] = []
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
            new_box = (start[0], start[1], end[0], end[1])
            self.boxes.append(new_box)
            self.selected_index = len(self.boxes) - 1
            self.manual_history.append(self.selected_index)

    def delete_selected(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.boxes):
            return
        deleted = self.selected_index
        del self.boxes[deleted]
        self.manual_history = [index for index in self.manual_history if index != deleted]
        self.manual_history = [index - 1 if index > deleted else index for index in self.manual_history]
        self.selected_index = None

    def undo_last_manual(self) -> None:
        while self.manual_history:
            index = self.manual_history.pop()
            if 0 <= index < len(self.boxes):
                del self.boxes[index]
                self.manual_history = [i - 1 if i > index else i for i in self.manual_history]
                self.selected_index = None
                return

    def clear_all(self) -> None:
        self.boxes.clear()
        self.manual_history.clear()
        self.selected_index = None

    def render(self):
        display = cv2.resize(self.image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        drag_box = None
        if self.drag_start is not None and self.drag_current is not None:
            drag_box = (self.drag_start[0], self.drag_start[1], self.drag_current[0], self.drag_current[1])
        draw_overlay(display, self.boxes, self.scale, self.selected_index, self.draft_count, drag_box)
        return display


def edit_boxes(image, boxes: list[Box]) -> tuple[str, list[Box]]:
    editor = BoxEditor(image, boxes)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(editor.width * editor.scale), int(editor.height * editor.scale))
    cv2.setMouseCallback(WINDOW_NAME, editor.mouse_callback)

    while True:
        cv2.imshow(WINDOW_NAME, editor.render())
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key == ord("a"):
            cv2.destroyWindow(WINDOW_NAME)
            return "approved", editor.boxes
        if key == ord("e"):
            cv2.destroyWindow(WINDOW_NAME)
            return "background", editor.boxes
        if key == ord("s"):
            cv2.destroyWindow(WINDOW_NAME)
            return "skipped", editor.boxes
        if key == ord("q"):
            cv2.destroyWindow(WINDOW_NAME)
            return "quit", editor.boxes
        if key == ord("d"):
            editor.delete_selected()
        elif key == ord("z"):
            editor.undo_last_manual()
        elif key == ord("c"):
            editor.clear_all()


def main() -> int:
    args = parse_args()
    approved_images_dir = args.dataset_root / "images"
    approved_labels_dir = args.dataset_root / "labels"
    approved_images_dir.mkdir(parents=True, exist_ok=True)
    approved_labels_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    image_paths = sorted(DRAFT_IMAGES_DIR.glob("*.jpg"))
    total = len(image_paths)
    statuses = set(args.review_status or ["unreviewed", "skipped"])
    print(f"[OUTPUT] Approved samples will be copied to {args.dataset_root}")
    print(f"[REVIEW] Including statuses: {', '.join(sorted(statuses))}")

    for image_path in image_paths:
        row = manifest.setdefault(image_path.stem, {"auto_persons": 0, "status": "unreviewed"})
        if row.get("status") not in statuses:
            continue

        label_path = DRAFT_LABELS_DIR / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            row["status"] = "skipped"
            save_manifest(manifest)
            continue

        height, width = image.shape[:2]
        boxes = load_yolo_boxes(label_path, width, height)
        reviewed = total - count_status(manifest, "unreviewed")
        approved = count_status(manifest, "approved")
        background = count_status(manifest, "background")
        skipped = count_status(manifest, "skipped")
        detected = row.get("detected_count", row.get("auto_persons", len(boxes)))
        print(f"[REVIEW] {image_path.name} | draft persons: {detected}")
        print(f"[PROGRESS] {reviewed}/{total} reviewed | {approved} approved | {background} background | {skipped} skipped")
        print("[KEYS] drag=draw | click=select | a=approve | e=background | s=skip | d=delete | z=undo | c=clear | q=quit")

        decision, corrected_boxes = edit_boxes(image, boxes)

        if decision == "quit":
            save_manifest(manifest)
            print("[REVIEW] Progress saved.")
            return 0
        if decision == "approved":
            approve_boxes(image_path, label_path, approved_images_dir, approved_labels_dir, corrected_boxes, width, height)
            row["status"] = "approved"
            row["approved_box_count"] = len(corrected_boxes)
        elif decision == "background":
            approve_background(image_path, label_path, approved_images_dir, approved_labels_dir)
            row["status"] = "background"
            row["approved_box_count"] = 0
        else:
            row["status"] = "skipped"

        save_manifest(manifest)

    approved = count_status(manifest, "approved")
    background = count_status(manifest, "background")
    skipped = count_status(manifest, "skipped")
    print(f"[REVIEW DONE] Approved: {approved} | Background: {background} | Skipped: {skipped}")
    print(f"Run scripts/prepare_dataset.py --dataset-root {args.dataset_root} to split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
