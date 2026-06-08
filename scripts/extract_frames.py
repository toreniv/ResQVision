
import cv2, pathlib, sys

VIDEO_NAME = 'soldiers_drill.mp4'  # שנה לשם הסרטון שלך
EVERY_N_FRAMES = 3  # 1 = כל פריים, 2 = כל פריים שני וכו'

video = cv2.VideoCapture(VIDEO_NAME)
if not video.isOpened():
    sys.exit(f'Could not open video: {VIDEO_NAME}')

fps = video.get(cv2.CAP_PROP_FPS)
total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Video: {fps:.1f}fps, {total} frames, {total/fps:.1f}s')

out_dir = pathlib.Path('frames')
out_dir.mkdir(exist_ok=True)

# מחק פריימים ישנים
for f in out_dir.glob('*.jpg'):
    f.unlink()

i, saved = 0, 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    if i % EVERY_N_FRAMES == 0:
        cv2.imwrite(str(out_dir / f'frame_{saved:04d}.jpg'), frame)
        saved += 1
    i += 1

video.release()
print(f'Saved {saved} frames to frames/')