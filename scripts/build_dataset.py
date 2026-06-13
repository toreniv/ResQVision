from __future__ import annotations

import argparse
import subprocess
import sys


def run_step(args: list[str]) -> None:
    print(f"[RUN] {' '.join(args)}")
    subprocess.run(args, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build draft dataset artifacts for manual review.")
    parser.add_argument("--videos", nargs="+", required=True, help="Video path(s) to process.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable
    run_step([python, "scripts/extract_frames.py", "--videos", *args.videos])
    run_step([python, "scripts/dedup_frames.py"])
    run_step([python, "scripts/auto_label.py"])
    print("[NEXT STEP] Review draft labels:")
    print("python scripts/review_dataset.py")
    print("Then run:")
    print("python scripts/finalize_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
