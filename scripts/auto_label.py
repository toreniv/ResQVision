import cv2, pathlib
from ultralytics import YOLO

PERSON_CLASSES = {'person', 'pedestrian', 'people'}

model = YOLO('models/drone_tactical_best.pt')
frames_dir = pathlib.Path('frames')
labels_dir = pathlib.Path('dataset/labels/train')
images_dir = pathlib.Path('dataset/images/train')
labels_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

for img_path in sorted(frames_dir.glob('*.jpg')):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    results = model(img, imgsz=1280, conf=0.08)[0]

    label_lines = []
    for box in results.boxes:
        if results.names[int(box.cls)] not in PERSON_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        label_lines.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

    cv2.imwrite(str(images_dir / img_path.name), img)
    (labels_dir / img_path.stem).with_suffix('.txt').write_text('\n'.join(label_lines))
    print(f'{img_path.name}: {len(label_lines)} persons labeled')

print('Done!')