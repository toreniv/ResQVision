"""
ResQVision — Tactical Fine-Tuning Pipeline
==========================================
Automates the full transfer-learning workflow on a custom drone dataset:

  1. Locate and extract dataset.zip
  2. Validate dataset structure (data.yaml, image/label folders, train/val splits)
  3. Download VisDrone base weights from HuggingFace (mshamrai/yolov8s-visdrone)
  4. Fine-tune using Ultralytics YOLO
  5. Deploy best.pt → models/drone_tactical_best.pt

Usage:
    python scripts/train_tactical.py [--epochs N] [--batch N] [--imgsz N]

Do NOT modify yolo_detect.py. This script only produces the weights file that
yolo_detect.py's DRONE_MODEL_CANDIDATES list already prioritises first.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution — works regardless of the CWD you launch from.
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_ZIP_PATH      = PROJECT_ROOT / "dataset.zip"
EXTRACT_DIR           = PROJECT_ROOT / "dataset_extracted"
DEPLOYED_MODEL_PATH   = PROJECT_ROOT / "models" / "drone_tactical_best.pt"
TRAIN_PROJECT         = str(PROJECT_ROOT / "runs" / "detect")
TRAIN_NAME            = "train_tactical"

HUGGINGFACE_REPO_ID   = "mshamrai/yolov8m-visdrone"
# Ordered preference list for weight filename selection.
PT_PREFERENCE_ORDER   = ["best.pt", "yolov8m-visdrone.pt", "yolov8m.pt"]

# Default training hyper-parameters (overridable via CLI).
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ  = 640
DEFAULT_BATCH  = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    width = max(len(text) + 4, 60)
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _info(msg: str) -> None:
    print(f"  [--]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [!!]  {msg}", file=sys.stderr)


def _fail(msg: str, exc: Exception | None = None) -> None:
    """Print an error and exit with a non-zero code."""
    print(f"\n  [FAIL] {msg}", file=sys.stderr)
    if exc:
        print(f"         Cause: {exc}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1 — Extract dataset.zip
# ---------------------------------------------------------------------------

def extract_dataset() -> None:
    _banner("STEP 1 — Extract dataset.zip")

    if not DATASET_ZIP_PATH.exists():
        _fail(
            f"dataset.zip not found at expected path:\n"
            f"         {DATASET_ZIP_PATH}\n\n"
            "         Use the ResQVision UI -> 'Export Training Dataset' to generate it,\n"
            "         or place a compatible dataset.zip at the project root."
        )

    _info(f"Found: {DATASET_ZIP_PATH}  ({DATASET_ZIP_PATH.stat().st_size / 1024:.1f} KB)")

    if EXTRACT_DIR.exists():
        _warn(f"Extraction directory already exists — clearing it: {EXTRACT_DIR}")
        try:
            shutil.rmtree(EXTRACT_DIR)
        except Exception as exc:
            _fail(f"Could not remove old extraction directory: {EXTRACT_DIR}", exc)

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as archive:
            archive.extractall(EXTRACT_DIR)
    except zipfile.BadZipFile as exc:
        _fail("dataset.zip is corrupt or is not a valid ZIP archive.", exc)
    except Exception as exc:
        _fail("Unexpected error while extracting dataset.zip.", exc)

    _ok(f"Extracted to: {EXTRACT_DIR}")


# ---------------------------------------------------------------------------
# Step 2 — Validate dataset structure
# ---------------------------------------------------------------------------

def validate_dataset() -> Path:
    """
    Recursively find data.yaml and confirm that train and val splits exist
    with at least one image each. Returns the resolved path to data.yaml.
    """
    _banner("STEP 2 — Validate dataset structure")

    yaml_candidates = list(EXTRACT_DIR.rglob("data.yaml"))

    if not yaml_candidates:
        _fail(
            "data.yaml not found anywhere inside the extracted dataset.\n"
            "         Expected structure:\n"
            "           dataset_extracted/\n"
            "             data.yaml\n"
            "             images/train/  images/val/\n"
            "             labels/train/  labels/val/"
        )

    if len(yaml_candidates) > 1:
        _warn(f"Multiple data.yaml files found — using the first: {yaml_candidates[0]}")

    data_yaml_path = yaml_candidates[0]
    dataset_root   = data_yaml_path.parent

    _ok(f"data.yaml located: {data_yaml_path}")
    _info(f"Dataset root inferred as: {dataset_root}")

    # ---- Print the detected dataset tree --------------------------------
    print("\n  Detected dataset layout:")
    for item in sorted(dataset_root.rglob("*")):
        relative = item.relative_to(dataset_root)
        depth    = len(relative.parts) - 1
        prefix   = "    " + ("  " * depth) + ("- " if item.is_file() else "+ ")
        size     = f"  ({item.stat().st_size} B)" if item.is_file() else ""
        print(f"{prefix}{item.name}{size}")

    # ---- Validate train and val splits ----------------------------------
    required_splits = ("train", "val")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for split in required_splits:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split

        if not images_dir.is_dir():
            _fail(
                f"Missing required split directory: {images_dir}\n"
                "         Cannot invent a train/val split — stopping.\n"
                "         Add more samples via the ResQVision UI and re-export."
            )

        if not labels_dir.is_dir():
            _fail(
                f"Missing required labels directory: {labels_dir}\n"
                "         Ensure every split has a matching labels/ folder."
            )

        image_files = [
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        label_files = [
            f for f in labels_dir.iterdir()
            if f.is_file() and f.suffix == ".txt"
        ]

        if not image_files:
            _fail(
                f"No image files found in: {images_dir}\n"
                f"         The '{split}' split is empty — stopping."
            )

        _ok(
            f"Split '{split}': {len(image_files)} image(s), {len(label_files)} label(s)"
        )

    _ok("Dataset structure is valid.")
    return data_yaml_path



# ---------------------------------------------------------------------------
# Step 3 — Download VisDrone base weights from HuggingFace
# ---------------------------------------------------------------------------

def download_base_weights() -> Path:
    """
    Downloads the mshamrai/yolov8s-visdrone repository snapshot and locates
    the best available .pt weights file. Returns the local path to that file.
    """
    _banner("STEP 3 — Download VisDrone base weights from HuggingFace")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _fail(
            "huggingface_hub is not installed.\n"
            "         Run:  pip install huggingface_hub"
        )

    _info(f"Downloading repo: {HUGGINGFACE_REPO_ID}")
    _info("This may take a moment on first run (weights are cached locally).")

    try:
        local_repo_dir = snapshot_download(
            repo_id=HUGGINGFACE_REPO_ID,
            ignore_patterns=["*.md", "*.txt", "*.yaml", "*.yml", "*.json"],
        )
    except Exception as exc:
        _fail(
            f"Failed to download from HuggingFace repo '{HUGGINGFACE_REPO_ID}'.\n"
            "         Check your internet connection or the repo ID.",
            exc,
        )

    local_repo_path = Path(local_repo_dir)
    _ok(f"Repo downloaded to: {local_repo_path}")

    # ---- Find the best .pt file in the downloaded snapshot --------------
    all_pt_files = list(local_repo_path.rglob("*.pt"))

    if not all_pt_files:
        _fail(
            f"No .pt weight files found in the downloaded repo snapshot.\n"
            f"         Searched: {local_repo_path}\n"
            "         The repo structure may have changed — check HuggingFace manually."
        )

    _info(f"Found {len(all_pt_files)} .pt file(s): {[f.name for f in all_pt_files]}")

    # Apply preference order.
    selected: Path | None = None
    for preferred_name in PT_PREFERENCE_ORDER:
        for candidate in all_pt_files:
            if candidate.name.lower() == preferred_name.lower():
                selected = candidate
                break
        if selected:
            break

    # If none of the preferred names matched, use the only file (or fail).
    if selected is None:
        if len(all_pt_files) == 1:
            selected = all_pt_files[0]
            _warn(
                f"No preferred filename matched ({PT_PREFERENCE_ORDER}). "
                f"Using the only available .pt: {selected.name}"
            )
        else:
            _fail(
                f"Multiple .pt files found but none match the preferred names "
                f"{PT_PREFERENCE_ORDER}.\n"
                f"         Available files: {[f.name for f in all_pt_files]}\n"
                "         Specify the exact file by editing PT_PREFERENCE_ORDER in this script."
            )

    _ok(f"Selected base weights: {selected}")
    return selected


# ---------------------------------------------------------------------------
# Step 4 — Fine-tune
# ---------------------------------------------------------------------------

def run_training(
    data_yaml_path: Path,
    weights_path: Path,
    epochs: int,
    imgsz: int,
    batch: int,
) -> Path:
    """
    Initialises Ultralytics YOLO with the downloaded weights and runs training.
    Returns the path to the Ultralytics output directory for this run.
    """
    _banner("STEP 4 — Fine-tune YOLO on tactical dataset")

    try:
        from ultralytics import YOLO
    except ImportError:
        _fail(
            "ultralytics is not installed.\n"
            "         Run:  pip install ultralytics"
        )

    try:
        import torch
    except ImportError:
        _fail(
            "torch is not installed.\n"
            "         Run:  pip install torch"
        )

    _info(f"Base weights : {weights_path}")
    _info(f"Dataset YAML : {data_yaml_path}")
    _info(f"Epochs       : {epochs}")
    _info(f"Image size   : {imgsz}")
    _info(f"Batch size   : {batch}")
    _info(f"Output dir   : {TRAIN_PROJECT}/{TRAIN_NAME}")
    
    cuda_available = torch.cuda.is_available()
    _info(f"CUDA status  : {'Available' if cuda_available else 'Not Available'}")
    
    if cuda_available:
        _info(f"CUDA version : {torch.version.cuda}")
        _info(f"GPU device   : {torch.cuda.get_device_name(0)}")
    else:
        _fail(
            "No local CUDA GPU detected. Google Colab is required for training.\n"
            "         Please download dataset.zip and use the Colab notebook."
        )

    print()

    try:
        model = YOLO(str(weights_path))
    except Exception as exc:
        _fail(
            f"Failed to initialise YOLO from weights: {weights_path}\n"
            "         The .pt file may be corrupt or incompatible with this "
            "ultralytics version.",
            exc,
        )

    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(data_yaml_path.parent)
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=TRAIN_PROJECT,
            name=TRAIN_NAME,
            exist_ok=True,
        )
    except Exception as exc:
        os.chdir(original_cwd)
        _fail(
            "Training failed with an unexpected error.\n"
            "         Check GPU memory, dataset integrity, and ultralytics version.",
            exc,
        )
    finally:
        os.chdir(original_cwd)

    # Derive the actual save directory from the results object if possible,
    # otherwise fall back to the conventional path.
    try:
        run_save_dir = Path(results.save_dir)
    except AttributeError:
        run_save_dir = PROJECT_ROOT / "runs" / "detect" / TRAIN_NAME

    _ok(f"Training complete. Run directory: {run_save_dir}")
    return run_save_dir


# ---------------------------------------------------------------------------
# Step 5 — Deploy best.pt
# ---------------------------------------------------------------------------

def deploy_weights(run_save_dir: Path) -> None:
    """
    Searches the Ultralytics run directory for best.pt and copies it to
    models/drone_tactical_best.pt. Does NOT assume a fixed sub-path.
    """
    _banner("STEP 5 — Deploy trained weights")

    _info(f"Searching for best.pt in: {run_save_dir}")

    # Ultralytics saves to <run_dir>/weights/best.pt, but the directory
    # layout can shift between versions — so we search recursively.
    best_candidates = list(run_save_dir.rglob("best.pt"))

    if not best_candidates:
        _fail(
            "best.pt was not found anywhere inside the training run directory.\n"
            f"         Searched: {run_save_dir}\n"
            "         Training may have ended before saving (e.g., too few epochs\n"
            "         or the dataset is too small). Check ultralytics logs above."
        )

    if len(best_candidates) > 1:
        _warn(
            f"Multiple best.pt files found — using the most recently modified one.\n"
            f"         Candidates: {best_candidates}"
        )
        best_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    source_best_pt = best_candidates[0]
    _ok(f"Found trained best.pt: {source_best_pt}")

    DEPLOYED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(source_best_pt, DEPLOYED_MODEL_PATH)
    except Exception as exc:
        _fail(
            f"Failed to copy best.pt to deployment path.\n"
            f"         Source      : {source_best_pt}\n"
            f"         Destination : {DEPLOYED_MODEL_PATH}",
            exc,
        )

    _ok(f"Deployed to: {DEPLOYED_MODEL_PATH}")
    print(
        f"\n  ✓ models/drone_tactical_best.pt is now the primary inference model.\n"
        f"    yolo_detect.py will automatically prefer it on the next run.\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResQVision — Tactical Fine-Tuning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Input image size (pixels, square).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Training batch size.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    _banner("ResQVision — Tactical Fine-Tuning Pipeline")
    _info(f"Project root : {PROJECT_ROOT}")
    _info(f"Dataset ZIP  : {DATASET_ZIP_PATH}")
    _info(f"Extract dir  : {EXTRACT_DIR}")
    _info(f"Deploy target: {DEPLOYED_MODEL_PATH}")

    # Step 1 — Extract
    extract_dataset()

    # Step 2 — Validate
    data_yaml_path = validate_dataset()

    # Step 3 — Download base weights
    weights_path = download_base_weights()

    # Step 4 — Train
    run_save_dir = run_training(
        data_yaml_path=data_yaml_path,
        weights_path=weights_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    # Step 5 — Deploy
    deploy_weights(run_save_dir)

    _banner("Pipeline Complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
