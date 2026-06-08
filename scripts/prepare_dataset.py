import pathlib, shutil, random

random.seed(42)
images = sorted(pathlib.Path('dataset/images/train').glob('*.jpg'))
random.shuffle(images)

val_count = max(2, len(images) // 5)  # 20% validation
val_images = images[:val_count]

val_img_dir = pathlib.Path('dataset/images/val')
val_lbl_dir = pathlib.Path('dataset/labels/val')
val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

for img_path in val_images:
    lbl_path = pathlib.Path('dataset/labels/train') / img_path.with_suffix('.txt').name
    shutil.move(str(img_path), val_img_dir / img_path.name)
    if lbl_path.exists():
        shutil.move(str(lbl_path), val_lbl_dir / lbl_path.name)

yaml_content = f"""path: {pathlib.Path('dataset').absolute().as_posix()}
train: images/train
val: images/val
nc: 1
names: ['person']
"""
pathlib.Path('dataset/data.yaml').write_text(yaml_content)
print(f'Train: {len(images)-val_count} | Val: {val_count}')
print('dataset/data.yaml created')