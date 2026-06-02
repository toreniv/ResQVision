# ResQVision

**GPU-Accelerated Battlefield Casualty Prioritization using CUDA Attention, Drone Vision, and ResQBand-Inspired Telemetry**

---

## Overview

ResQVision is a research-oriented simulation platform for real-time battlefield casualty prioritization.

The project explores how modern AI attention mechanisms can be accelerated using NVIDIA CUDA and applied to battlefield medical triage scenarios.

A simulated drone observes the battlefield while soldiers continuously transmit physiological telemetry inspired by the ResQBand concept. The system processes this information using a CUDA-accelerated scaled dot-product attention engine and generates evacuation priorities in real time.

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

---

## System Architecture

```text
Drone Observation Layer
            │
            ▼
Soldier Detection / Tracking
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
```

---

## CUDA Components

Current CUDA kernels:

1. QKᵀ Matrix Multiplication
2. Attention Scaling
3. Row-wise Softmax
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

Planned optimization stages:

### Phase 1 - Baseline CUDA

* Global memory implementation
* Separate kernels
* Functional correctness validation

### Phase 2 - Shared Memory Optimization

* Tiled matrix multiplication
* Reduced global memory access
* Improved cache utilization

### Phase 3 - Advanced Optimizations

* Kernel fusion
* Memory coalescing improvements
* Occupancy tuning
* Larger battlefield simulations

### Phase 4 - Real-Time Processing

* Continuous telemetry streams
* Live drone observations
* Interactive battlefield command dashboard

---

## Benchmark Goals

The project evaluates performance for:

| Soldiers | Attention Dimension |
| -------- | ------------------- |
| 128      | 64                  |
| 256      | 64                  |
| 512      | 64                  |
| 1024     | 64                  |

Metrics:

* CPU execution time
* GPU execution time
* Speedup factor
* Numerical correctness

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
├── README.md
│
├── docs/
│
├── outputs/
│   ├── benchmark_results.csv
│   ├── risk_ranking.csv
│   ├── attention_stats.csv
│   └── attention_heatmap.csv
│
└── .gitignore
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

* benchmark_results.csv
* risk_ranking.csv
* attention_stats.csv
* attention_heatmap.csv

---

## Future Work

### Computer Vision

* YOLO-based soldier detection
* Multi-object tracking
* Casualty localization

### Autonomous Drone Support

* GPS-denied navigation
* Visual Odometry
* SLAM-based positioning

### Battlefield Command Center

* Real-time dashboard
* Drone command view
* Interactive evacuation planning

### ResQBand Integration

* Real telemetry ingestion
* LoRa communication layer
* Wearable sensor network simulation

---

## Academic Context

This project was developed as part of a GPU Programming / CUDA course and focuses on applying parallel computing techniques to a realistic defense and emergency-response scenario.

---

## Author

Niv Toren

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

Example output files:

```text
benchmark_results.csv
risk_ranking.csv
attention_stats.csv
attention_heatmap.csv
```

These outputs are used to evaluate both computational performance and battlefield decision-support quality.
---