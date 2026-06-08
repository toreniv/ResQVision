# Requirements: pip install ultralytics opencv-python

from datetime import datetime, timezone
import json
import pathlib
import time

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics opencv-python")
    raise SystemExit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] pip install ultralytics opencv-python")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
out_dir = pathlib.Path("frontend/public/data")
out_dir.mkdir(parents=True, exist_ok=True)

detections_path = out_dir / "detections.json"
preview_path    = out_dir / "detection_preview.jpg"

MODEL_CANDIDATES = (
    pathlib.Path("models/drone_tactical_best.pt"),
    pathlib.Path("yolov8s.pt"),
    pathlib.Path("yolov8n.pt"),
)


def load_preferred_model():
    last_error = None
    for model_path in MODEL_CANDIDATES:
        try:
            model = YOLO(str(model_path))
            print(f"[ResQVision LIVE] Model: {model_path}")
            return model
        except Exception as exc:
            last_error = exc
    raise last_error

# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = load_preferred_model()

print("[ResQVision LIVE] Starting. Press 'q' in the preview window to quit.")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame – retrying …")
            time.sleep(0.1)
            continue

        frame_height, frame_width = frame.shape[:2]

        # --- Inference ---
        results = model(frame, verbose=False)
        result  = results[0]

        # --- Filter persons ---
        detections = []
        detection_id = 1

        for box in result.boxes:
            cls_name = result.names[int(box.cls.item())]
            if cls_name != "person":
                continue

            confidence = round(float(box.conf.item()), 2)

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            w = x2 - x1
            h = y2 - y1
            center = [x1 + w // 2, y1 + h // 2]

            detections.append({
                "id":         detection_id,
                "class":      "person",
                "confidence": confidence,
                "bbox":       [x1, y1, w, h],
                "center":     center,
            })
            detection_id += 1

        payload = {
            "source": "live_camera",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "detections": detections,
        }

        # --- Atomic write: detections.json ---
        tmp_json = detections_path.with_suffix(".tmp")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        tmp_json.replace(detections_path)

        # --- Annotated frame ---
        annotated = result.plot()

        # --- Atomic write: detection_preview.jpg ---
        tmp_img = out_dir / "detection_preview.tmp.jpg"
        cv2.imwrite(str(tmp_img), annotated)
        tmp_img.replace(preview_path)

        # --- Display ---
        cv2.imshow("ResQVision LIVE", annotated)

        # --- Console summary ---
        print(f"[ResQVision LIVE] Detected {len(detections)} person(s)")

        # --- Throttle & quit ---
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[ResQVision LIVE] Stopped.")
