import cv2, pathlib

VIDEO_NAME = 'soldiers_drill.mp4'  # שנה לשם הסרטון שלך

video = cv2.VideoCapture(VIDEO_NAME)
fps = video.get(cv2.CAP_PROP_FPS)
interval = int(fps * 0.5)  # פריים כל חצי שנייה במקום כל 2 שניות
pathlib.Path('frames').mkdir(exist_ok=True)
i, saved = 0, 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    if i % interval == 0:
        cv2.imwrite(f'frames/frame_{saved:04d}.jpg', frame)
        saved += 1
    i += 1

video.release()
print(f'Saved {saved} frames to frames/')