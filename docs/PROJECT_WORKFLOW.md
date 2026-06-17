# ResQVision Project Workflow

## A. System Overview

ResQVision is a CUDA-accelerated tactical triage demo for a university project prototype. The academic core is `resqvision.cu`, which implements Scaled Dot-Product Attention in CUDA and exports benchmark, correctness, and risk-priority artifacts.

The frontend is a React dashboard that visualizes the generated data as benchmark analytics, risk ranking, and a tactical map. The computer vision layer is optional: it demonstrates UAV-style visual detection using image upload or camera input and YOLO-style inference when the local model/server dependencies are available.

This project should be presented as a simulation and demo, not as a real operational battlefield tool. Use framing such as "simulation", "demo", "visual detection layer", "UAV-style input", and "human-reviewed preview".

## B. Execution Modes

### Recommended One-Command Demo

Run this from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_resqvision_demo.ps1
```

Useful variants:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_resqvision_demo.ps1 -StartYolo
powershell -ExecutionPolicy Bypass -File .\scripts\start_resqvision_demo.ps1 -NoYolo
powershell -ExecutionPolicy Bypass -File .\scripts\start_resqvision_demo.ps1 -UseColabImport
powershell -ExecutionPolicy Bypass -File .\scripts\status_resqvision_demo.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stop_resqvision_demo.ps1
```

`start_resqvision_demo.ps1` is the recommended launcher for the live demo. It prepares or reuses the Python environment, runs or imports CUDA outputs, validates dashboard JSON, starts the optional YOLO backend when possible, starts the React dashboard, opens the browser, and prints status.

`run_cuda_local.ps1` remains the local CUDA-only runner. `import_colab_outputs.ps1` remains the Colab fallback importer. `yolo_server.py` is optional and may be started by the launcher when dependencies are available. The dashboard still works if YOLO is offline.

### 1. Local CUDA Mode

Use this mode when the machine has:

- NVIDIA GPU
- NVIDIA driver
- CUDA Toolkit / `nvcc`
- MSVC `cl` compiler on Windows

Flow:

1. Run `scripts/check_cuda.ps1`.
2. Run `scripts/run_cuda_local.ps1`.
3. Generate CUDA benchmark CSV files.
4. Convert CSV outputs to JSON with `scripts/csv_to_json.py` or the local pipeline script.
5. Open the frontend and confirm it reads the JSON from `frontend/public/data`.

The CUDA benchmark output keeps `GPU_time_ms` as a legacy-compatible field that maps to the tiled CUDA result. In the UI, this value is displayed as "CUDA Tiled ms" for clarity.

Safe benchmark wording for demos:

```text
Up to 213x CUDA Tiled speedup on Google Colab Tesla T4, with correctness PASS and Top-10 overlap 10/10. Performance depends on GPU, driver, CUDA runtime, and workload size.
```

### 2. Colab Fallback Mode

Use this mode when local CUDA is unavailable.

Flow:

1. Open `ResQVision_Colab_Workflow.ipynb`.
2. Enable a GPU runtime.
3. Upload the latest `resqvision.cu`.
4. Run the CUDA benchmark.
5. Run the stability sweep.
6. Export `resqvision_cuda_outputs.zip`.
7. Put the ZIP in `Downloads`.
8. Run `scripts/import_colab_outputs.ps1` locally.
9. Open the frontend and confirm it reads the imported JSON.

This is the recommended fallback for laptops or desktops without the full NVIDIA CUDA toolchain.

Important: do not run `scripts/csv_to_json.py` immediately after importing Colab outputs unless you intentionally want to regenerate JSON from local CSV files. It may overwrite newer Colab JSON with older local CSV data.

### 3. Computer Vision Demo Mode

Use this mode to simulate UAV/drone visual input.

Flow:

1. Upload an image or use browser camera/webcam input as a UAV-style demo feed.
2. Run detection if the YOLO model and local backend dependencies are available.
3. Generate `detections.json`, `tactical_fusion.json`, or equivalent dashboard data.
4. Show visual detection results in the dashboard.
5. If model confidence is limited or the backend is unavailable, use the human-reviewed demo preview for the final presentation.

This mode is a visual detection layer for the demo. It should remain human-reviewed and clearly labeled as simulation output.

## C. Website Flow

The frontend reads these files from `frontend/public/data`:

- `benchmark_results.json`: CUDA benchmark rows, including CPU time, CUDA tiled time, speedup, correctness, error metrics, and Top-10 overlap.
- `risk_ranking.json`: fallback triage priority rows when no tactical fusion file is present.
- `attention_stats.json`: optional correctness summary; if absent, the frontend can derive correctness display fields from `benchmark_results.json`.
- `tactical_fusion.json`: optional fused tactical output from visual detections and triage data.
- `detections.json`: optional raw or live YOLO-style detection output.
- `human_review_detections.json`: optional human-reviewed detection output for demo preview.

When `tactical_fusion.json` is present and contains targets, the dashboard uses it for the tactical view. Otherwise, it falls back to `risk_ranking.json`.

## D. What Is Real Now vs Simulated

What is real now:

- CUDA benchmark output.
- CPU/GPU correctness validation.
- CUDA Basic + CUDA Tiled comparison.
- Risk ranking JSON.
- Attention statistics JSON.
- React analytics charts.
- Tactical command top targets.
- Human-reviewed visual preview.

What is simulated or optional:

- Live UAV feed.
- Real-time YOLO backend unless `scripts/yolo_server.py` is running.
- ResQBand live telemetry.
- Real clinical decision approval.
- Fully automated evacuation dispatch.

## E. Final Demo Flow

Suggested live demo order:

1. Analytics:
   - Show CUDA Tiled ms.
   - Show speedup.
   - Show correctness PASS.
   - Show Top-10 overlap 10/10.
2. Tactical Command:
   - Show top evacuation targets.
   - Show tactical map markers.
   - Show recommended actions.
3. System Architecture:
   - Explain `resqvision.cu` to JSON to React dashboard.
   - Distinguish real CUDA data from simulated visual/demo layers.
4. Computer Vision:
   - Show human-reviewed demo preview.
   - Show optional upload/camera flow.
   - Explain that YOLO inference requires the local backend.
5. Explain fallback:
   - Use local GPU mode when available.
   - Use Colab fallback mode otherwise.

## Optional Local GPU / Image Detection Support

The project currently includes these computer vision scripts:

- `scripts/yolo_detect.py`
- `scripts/yolo_live.py`
- `scripts/yolo_server.py`
- `scripts/fuse_yolo_to_tactical.py`
- `scripts/demo_annotate_preview.py`

Expected local GPU computer vision flow:

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Check CUDA/GPU:
   ```powershell
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Run the local YOLO server:
   ```powershell
   python scripts/yolo_server.py
   ```
   If using the repository virtual environment on Windows:
   ```powershell
   venv\Scripts\python.exe scripts\yolo_server.py
   ```
4. Open the frontend:
   ```powershell
   cd frontend
   npm run dev
   ```
5. Use the dashboard:
   - Upload an image or use camera input on the Computer Vision page.
   - Run YOLO if the backend is running.
   - Review or add manual tactical tags if needed.
   - View refreshed tactical fusion results in the dashboard.

The existing local backend uses endpoints such as `POST /api/yolo/upload`, `POST /api/yolo/upload_and_save`, `POST /api/markers/save`, and `POST /api/dataset/export` on `http://127.0.0.1:8000`. A generic `POST /api/detect-image` endpoint is not currently required by the frontend.

If the backend is not running, the frontend should fail gracefully and show the human-reviewed demo preview.

## Execution Modes

### Local GPU Mode

Local GPU execution requires:

- NVIDIA GPU
- NVIDIA driver
- CUDA Toolkit / `nvcc`
- MSVC Build Tools on Windows

Run:

```powershell
scripts/check_cuda.ps1
scripts/run_cuda_local.ps1
```

### Colab Fallback Mode

Use this when local CUDA is unavailable.

Run `ResQVision_Colab_Workflow.ipynb` with GPU runtime, download `resqvision_cuda_outputs.zip`, and import locally with:

```powershell
scripts/import_colab_outputs.ps1
```

### Computer Vision Local Mode

For image/camera YOLO inference, run:

```powershell
python scripts/yolo_server.py
```

Then open the frontend and use the Computer Vision page.

If unavailable:

- Use the Google Colab workflow.
- Import `resqvision_cuda_outputs.zip` with `scripts/import_colab_outputs.ps1`.
