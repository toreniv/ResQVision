from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"


def load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def status_line(name: str, status: str, detail: str = "") -> None:
    print(f"{name:<32} {status:<8} {detail}")


def first_benchmark_row(data: Any) -> dict[str, Any] | None:
    if isinstance(data, list) and data and isinstance(data[-1], dict):
        return data[-1]
    return None


def number_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    print("ResQVision Dashboard Data Validation")
    print("=" * 48)

    benchmark_path = DATA_DIR / "benchmark_results.json"
    risk_path = DATA_DIR / "risk_ranking.json"
    attention_path = DATA_DIR / "attention_stats.json"
    fusion_path = DATA_DIR / "tactical_fusion.json"
    human_review_path = DATA_DIR / "human_review_detections.json"

    failures = 0

    benchmark = load_json(benchmark_path)
    if benchmark is None:
        status_line("benchmark_results.json", "FAIL", "missing or invalid")
        failures += 1
    else:
        rows = len(benchmark) if isinstance(benchmark, list) else 0
        status_line("benchmark_results.json", "OK", f"{rows} rows")

    risk = load_json(risk_path)
    if risk is None:
        status_line("risk_ranking.json", "FAIL", "missing or invalid")
        failures += 1
    else:
        rows = len(risk) if isinstance(risk, list) else 0
        status_line("risk_ranking.json", "OK", f"{rows} rows")

    attention = load_json(attention_path)
    status_line("attention_stats.json", "OK" if attention is not None else "WARN", "optional correctness summary" if attention is not None else "missing")

    row = first_benchmark_row(benchmark)
    if row is None:
        status_line("benchmark schema", "FAIL", "expected array of rows")
        failures += 1
    else:
        required_any = {
            "CPU time": ("CPU_time_ms", "cpu", "cpu_ms"),
            "CUDA tiled time": ("GPU_tiled_time_ms", "GPU_time_ms", "gpu", "gpu_ms"),
            "speedup": ("speedup_tiled", "speedup", "Speedup"),
            "correctness": ("tiled_correctness", "correctness", "status"),
        }
        for label, keys in required_any.items():
            present = any(key in row and row.get(key) not in (None, "") for key in keys)
            status_line(label, "OK" if present else "FAIL", " / ".join(keys))
            if not present:
                failures += 1

        speedup = number_or_none(row.get("speedup_tiled", row.get("speedup")))
        if speedup is not None:
            rounded = round(speedup)
            detail = f"max row reports {speedup:.3f}x; UI may display up to {rounded}x"
            if abs(speedup - 213.061) <= 1.0:
                detail += " (Colab Tesla T4 reference)"
            status_line("speedup display", "OK", detail)
        else:
            status_line("speedup display", "WARN", "speedup_tiled/speedup not numeric")

    fusion = load_json(fusion_path)
    status_line("tactical_fusion.json", "OK" if fusion is not None else "WARN", "optional; dashboard falls back to risk_ranking.json" if fusion is None else "present")

    human = load_json(human_review_path)
    status_line("human_review_detections.json", "OK" if human is not None else "WARN", "optional CV demo preview" if human is not None else "missing; CV preview may fall back")

    print("=" * 48)
    if failures:
        print(f"Validation complete with {failures} required issue(s).")
        return 1

    print("Validation complete. Required dashboard data is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
