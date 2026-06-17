from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.stdout = (LOG_DIR / "yolo_server.out.log").open("w", encoding="utf-8", buffering=1)
sys.stderr = (LOG_DIR / "yolo_server.err.log").open("w", encoding="utf-8", buffering=1)

from yolo_server import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
