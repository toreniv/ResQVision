# Requirements: pip install ultralytics opencv-python

import argparse
from datetime import datetime, timezone
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_IMAGE = "scripts/sample_input.jpg"

parser = argparse.ArgumentParser(
    description="Run YOLOv8n on an image and export person detections for ResQVision."
)
parser.add_argument(
    "--image",
    default=DEFAULT_IMAGE,
    help=f"Path to input image (default: {DEFAULT_IMAGE})",
)
args = parser.parse_args()

image_path = pathlib.Path(args.image)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
if not image_path.exists():
    print(f"[ERROR] Image not found: {image_path}")
    sys.exit(1)

print(f"[ResQVision] Input: {image_path}")

image = cv2.imread(str(image_path))
if image is None:
    print(f"[ERROR] Could not read image: {image_path}")
    sys.exit(1)

frame_height, frame_width = image.shape[:2]

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
out_dir = pathlib.Path("frontend/public/data")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Run YOLOv8n
# ---------------------------------------------------------------------------
model = YOLO("yolov8n.pt")          # downloads weights on first run
results = model(str(image_path))    # list of Results objects

result = results[0]

# ---------------------------------------------------------------------------
# Filter persons only (class id 0 in COCO)
# ---------------------------------------------------------------------------
detections = []
detection_id = 1

for box in result.boxes:
    cls_name = result.names[int(box.cls.item())]
    if cls_name != "person":
        continue

    confidence = round(float(box.conf.item()), 2)

    # box.xywh: [cx, cy, w, h] in pixel coords
    # We want top-left x, y so convert from xyxy
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
    w = x2 - x1
    h = y2 - y1

    detections.append({
        "id": detection_id,
        "class": "person",
        "confidence": confidence,
        "bbox": [x1, y1, w, h],
    })
    detection_id += 1

# ---------------------------------------------------------------------------
# Save detections.json
# ---------------------------------------------------------------------------
payload = {
    "source": "offline_image",
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "frame_width": frame_width,
    "frame_height": frame_height,
    "detections": detections,
}

json_path = out_dir / "detections.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

if detections:
    print(f"[ResQVision] Detected {len(detections)} person(s)")
else:
    print("[ResQVision] No persons detected. JSON saved as empty array.")

print(f"[ResQVision] Saved → {json_path}")

# ---------------------------------------------------------------------------
# Save annotated preview image
# ---------------------------------------------------------------------------
annotated = result.plot()           # returns BGR NumPy array
preview_path = out_dir / "detection_preview.jpg"
cv2.imwrite(str(preview_path), annotated)
print(f"[ResQVision] Saved → {preview_path}")
