# ResQVision

**GPU-Accelerated Battlefield Casualty Prioritization using CUDA Attention, Drone Vision, and ResQBand-Inspired Telemetry**

---

## Overview

ResQVision is a research-oriented simulation platform for real-time battlefield casualty prioritization.

The project explores how modern AI attention mechanisms can be accelerated using NVIDIA CUDA and applied to battlefield medical triage scenarios.

A simulated drone observes the battlefield while soldiers continuously transmit physiological telemetry inspired by the ResQBand concept. The system processes this information using a CUDA-accelerated scaled dot-product attention engine and generates evacuation priorities in real time.

> *"ResQVision does not only rank casualties. It converts GPU-computed risk into operational recommendations."*

ResQVision includes:
* CUDA Attention Engine
* JSON export pipeline
* React tactical dashboard
* Attention visualization layer
* Rule-based operational recommendations
* YOLO computer vision integration

The dashboard consumes CUDA-generated outputs and visualizes battlefield decision-support information.

---

## Quick Start

> Run from the **project root** (`ResQVision/`) for all commands below.

### One-time setup (Windows)

```powershell
.\setup.ps1
```

This creates a Python virtual environment, installs all dependencies from `requirements.txt`, and runs `npm install` inside `frontend/`.

---

### Frontend

```bash
cd frontend
npm install      # skip if setup.ps1 was already run
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

### YOLO Detection (Offline)

```bash
# Activate the virtual environment first
venv\Scripts\activate

python scripts/yolo_detect.py --image scripts/sample_input.png
```

Outputs written to `frontend/public/data/`:

| File | Description |
|---|---|
| `detections.json` | Per-person detection results |
| `detection_preview.jpg` | Annotated frame with bounding boxes |

The Computer Vision page in the dashboard picks these up automatically.

---

### YOLO Live Detection

```bash
venv\Scripts\activate
python scripts/yolo_live.py
```

Continuously writes `detections.json` and `detection_preview.jpg` from webcam frames. The dashboard polls for updates every second — no backend required.

---

### CUDA

```bash
nvcc -O2 resqvision.cu -o resqvision
./resqvision
```

Then export results to JSON:

```bash
python scripts/csv_to_json.py
```

---

### JSON Integration

After running the CUDA binary or YOLO script, generated JSON files live in:

```
frontend/public/data/
├── benchmark_results.json
├── risk_ranking.json
├── attention_stats.json
└── detections.json
```

The dashboard loads these files on startup. If any file is missing or malformed, it falls back to built-in mock data automatically — so the demo always works.

### Export from Colab

Run the last cell in `ResQVision_Colab_Workflow.ipynb` to download `resqvision_cuda_outputs.zip`, then run locally (PowerShell):

```powershell
Remove-Item ".\temp_exports" -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive "$env:USERPROFILE\Downloads\resqvision_cuda_outputs.zip" `
  -DestinationPath ".\temp_exports" -Force
Copy-Item ".\temp_exports\benchmark_results.json" ".\frontend\public\data\" -Force
Copy-Item ".\temp_exports\risk_ranking.json"       ".\frontend\public\data\" -Force
Copy-Item ".\temp_exports\attention_stats.json"    ".\frontend\public\data\" -Force
```

---

## Project Motivation

Modern battlefield environments generate large volumes of information from sensors, wearable devices, drones, and communication systems.

Medical teams and commanders must rapidly identify the most critical casualties and allocate evacuation resources efficiently.

Traditional sequential processing may become a bottleneck as the number of monitored soldiers increases.

ResQVision investigates how GPU-accelerated attention mechanisms can process battlefield telemetry in parallel and provide real-time casualty prioritization for large-scale scenarios.

The project combines concepts from:

* Parallel Computing
* CUDA Programming
* Artificial Intelligence
* Medical Decision Support
* Defense Technology
* Computer Vision

to create a realistic simulation of next-generation battlefield triage systems.

---

## Key Features

### CUDA-Accelerated Attention Engine

* Scaled Dot-Product Attention
* GPU implementation in CUDA C++
* Parallel matrix operations
* Softmax computation on GPU
* CPU vs GPU performance comparison

### Synthetic ResQBand Telemetry

Each simulated soldier generates:

* Heart Rate (HR)
* Blood Oxygen Saturation (SpO₂)
* Body Temperature
* Respiration Rate
* Motion Level
* Signal Quality
* Battery Status
* Battlefield Coordinates

### Casualty Risk Assessment

The system computes:

* Medical risk score
* Attention-based contextual relevance
* Evacuation priority ranking

### Visualization

* Attention Heatmap
* Battlefield Risk Map
* Top Evacuation Targets
* Benchmark Performance Graphs
* YOLO Detection Preview

---

## Frontend Dashboard

### Mission Plan
* Pre-operation planning dashboard
* Mission readiness overview
* Priority casualty preview
* Operational map

### Tactical Command
* Live casualty ranking
* Tactical map
* Attention halo visualization
* Recommended Actions panel
* UAV routing indicators

### Analytics
* CPU vs GPU benchmark charts
* Speedup analysis
* Correctness validation
* Performance metrics

### System Architecture
* End-to-end pipeline visualization
* Simulated vs real components
* Data flow overview

### Computer Vision
* YOLO person detection results
* Confidence scores per detection
* Bounding box coordinates
* Live refresh indicator (● LIVE) when webcam script is running
* Automatic fallback to mock data when no detection file is present

---

## JSON Integration Workflow

```
CUDA Output
↓
CSV Files
↓
JSON Export (scripts/csv_to_json.py)
↓
frontend/public/data/
↓
React Dashboard
```

Artifacts include:
* `benchmark_results.json`
* `risk_ranking.json`
* `attention_stats.json`

**Fallback behavior:**
If JSON files are unavailable, the dashboard automatically falls back to mock data.

---

## Attention Visualization Layer

`attention_stats.json` contains per-soldier attention values:
* `soldier_id`
* `max_attention`
* `mean_attention`
* `entropy`

Visualization tiers on the tactical map:
* Top 3 attention targets → 🔴 red halo
* Next 3 attention targets → 🟠 orange halo
* Remaining targets → 🔵 blue halo

The visualization is derived directly from CUDA attention outputs — no manual annotation.

---

## Operational Decision Support

The dashboard features a Recommended Action Engine that derives actions from:
* Risk ranking
* Casualty category
* Physiological status

The engine is implemented as a **pure function** (`deriveRecommendedActions`) — no backend, no ML model, deterministic and testable.

Example output:
```
1 · Evacuate Soldier 388       Risk 98.2 · HR 180 bpm · SpO₂ 74%
2 · Dispatch Trauma Team Bravo  3 critical casualties in sector
3 · Route UAV-1 to Cluster     Top 3 targets in operational range
4 · Monitor Soldier 282        HR 168 bpm · trend watch
```

**Note:** This is a rule-based prototype and not a clinical decision system.

---

## System Architecture

```text
Drone Observation Layer
            │
            ▼
Soldier Detection / Tracking (YOLO)
            │
            ▼
ResQBand Telemetry Stream
            │
            ▼
Feature Matrix Generation
            │
            ▼
CUDA Attention Engine
            │
            ▼
Risk Assessment
            │
            ▼
Evacuation Priority Ranking
            │
            ▼
Recommended Actions → Tactical Dashboard
```

---

## CUDA Components

Current CUDA kernels:

1. QKᵀ Matrix Multiplication
2. Attention Scaling (1/√d)
3. Row-wise Softmax (numerically stable)
4. Attention × V Computation

The implementation demonstrates:

* Thread-to-data mapping
* Grid and block configuration
* Global memory operations
* Numerical stability in Softmax
* CPU/GPU correctness validation

Future versions will introduce:

* Shared Memory Tiling
* Optimized Matrix Multiplication
* Kernel Fusion
* Larger-scale battlefield simulations

---

## Performance Optimization Roadmap

Current implementation focuses on correctness and baseline CUDA execution.

### Phase 1 – Baseline CUDA
* Global memory implementation
* Separate kernels
* Functional correctness validation

### Phase 2 – Shared Memory Optimization
* Tiled matrix multiplication
* Reduced global memory access
* Improved cache utilization

### Phase 3 – Advanced Optimizations
* Kernel fusion
* Memory coalescing improvements
* Occupancy tuning
* Larger battlefield simulations

### Phase 4 – Real-Time Processing
* Continuous telemetry streams
* Live drone observations
* Interactive battlefield command dashboard

---

## Benchmark Goals

| Soldiers | Attention Dimension |
|---|---|
| 128 | 64 |
| 256 | 64 |
| 512 | 64 |
| 1024 | 64 |

Metrics:
* CPU execution time
* GPU execution time
* Speedup factor
* Numerical correctness

---

## Current Demonstrated Results

* **49× GPU acceleration** (512 soldiers benchmark)
* Successful CPU/GPU correctness validation
* Top-10 overlap validation
* Attention-based casualty prioritization
* Tactical map with attention halo visualization
* Rule-based Recommended Action Engine
* YOLO computer vision integration

---

## Expected Results

The project aims to demonstrate:

* Correct numerical agreement between CPU and GPU implementations.
* Significant execution-time reduction using CUDA acceleration.
* Real-time prioritization of hundreds to thousands of simulated soldiers.
* Scalability as battlefield size increases.
* Clear visualization of casualty risk and evacuation priorities.

Success criteria include:

* CPU/GPU correctness validation
* Stable attention computation
* Measurable GPU speedup
* Reproducible benchmark results

---

## Current Repository Structure

```text
ResQVision/
│
├── resqvision.cu
├── ResQVision_Colab_Workflow.ipynb
├── setup.ps1
├── requirements.txt
├── README.md
│
├── scripts/
│   ├── yolo_detect.py
│   ├── yolo_live.py
│   └── csv_to_json.py
│
├── outputs/
│   ├── benchmark_results.csv
│   ├── risk_ranking.csv
│   ├── attention_stats.csv
│   └── attention_heatmap.csv
│
├── frontend/
│   ├── public/data/
│   │   ├── benchmark_results.json
│   │   ├── risk_ranking.json
│   │   ├── attention_stats.json
│   │   └── detections.json
│   └── src/
│       ├── App.jsx
│       └── styles.css
│
└── docs/
```

---

## Running in Google Colab

Compile:

```bash
nvcc -O2 resqvision.cu -o resqvision
```

Run:

```bash
./resqvision
```

Generated outputs:

* `benchmark_results.csv`
* `risk_ranking.csv`
* `attention_stats.csv`
* `attention_heatmap.csv`

Export all artifacts as ZIP — run the last cell in `ResQVision_Colab_Workflow.ipynb`.

---

## Future Work

### Computer Vision
* Multi-object tracking
* Casualty localization on tactical map
* YOLO + risk ranking fusion

### Autonomous Drone Support
* GPS-denied navigation
* Visual Odometry
* SLAM-based positioning
* Live UAV integration

### Battlefield Command Center
* Real-time dashboard updates
* Drone command view
* Interactive evacuation planning
* Direct CUDA-to-dashboard streaming

### ResQBand Integration
* Real telemetry ingestion
* LoRa communication layer
* Wearable sensor network simulation
* ResQBand hardware integration

---

## Academic Context

This project was developed as part of a GPU Programming / CUDA course and focuses on applying parallel computing techniques to a realistic defense and emergency-response scenario.

---

## Author

**Niv Toren**  
B.Sc. Electrical Engineering

Areas of Interest:
- Embedded Systems
- CUDA Programming
- Computer Vision
- Artificial Intelligence
- Defense Technology
- Medical Wearable Systems

---

## Example Outputs

The project generates:

- Attention Heatmaps
- Battlefield Risk Maps
- Evacuation Priority Rankings
- CPU vs GPU Benchmark Reports
- YOLO Detection Previews with bounding boxes

Example output files:

```text
benchmark_results.csv
risk_ranking.csv
attention_stats.csv
attention_heatmap.csv
```

These outputs are used to evaluate both computational performance and battlefield decision-support quality.
```

---

## YOLO Detection Preview

![YOLO detection preview](docs/images/yolo_detection_preview.jpg)

---



***

