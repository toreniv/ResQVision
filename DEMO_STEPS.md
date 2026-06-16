# ResQVision Demo Steps

## Main Presentation Flow

Start from the CUDA requirement. The dashboard and YOLO layer come after the GPU implementation has been explained.

### 1. Problem

Open with the computational problem:

```text
Many simulated soldiers transmit telemetry at the same time.
The system must rank casualty priority quickly enough for an operational dashboard.
```

Explain that `N` is the number of simulated soldiers and `d = 64` is the attention feature dimension. The project uses attention because each soldier's risk can be evaluated in context with the rest of the battlefield state.

### 2. Attention Formula

Show the Scaled Dot-Product Attention formula:

```text
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d)) * V
```

Connect the math to the project:

* `Q`, `K`, and `V` are generated from simulated ResQBand-style battlefield telemetry.
* The attention output feeds casualty prioritization.
* The CPU version is the correctness reference.
* The CUDA versions are the parallel implementations being evaluated.

### 3. CUDA Kernels

Open `resqvision.cu` and show that the attention pipeline is decomposed into kernels:

1. `Q * K^T`
2. Scaling by `1 / sqrt(d)`
3. Row-wise numerically stable Softmax
4. `Attention * V`

Then show that the project compares three paths:

1. CPU reference
2. CUDA basic
3. CUDA tiled

Important kernel names to point at:

```text
qk_transpose_kernel
qk_transpose_tiled_kernel
scale_kernel
row_softmax_kernel
attention_v_kernel
attention_v_tiled_kernel
```

The tiled CUDA path uses shared memory for both matrix multiplication stages: `Q * K^T` and `Attention * V`.

### 4. Thread/Block Mapping

Show the matrix thread mapping:

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

Explain:

* The grid is two-dimensional because the output matrices are two-dimensional.
* Each valid CUDA thread computes one output matrix element.
* Boundary checks prevent invalid memory access when dimensions do not divide evenly by the block size.

For the tiled implementation, show:

```text
TILE_WIDTH = 16
16 x 16 threads per block
```

Each block cooperatively loads a tile into shared memory, synchronizes, reuses the tile data, and accumulates partial dot products.

### 5. Benchmark

Run or show:

```powershell
.\scripts\run_cuda_local.ps1
```

If local CUDA is unavailable, use the Colab output or `outputs/benchmark_results.csv`.

Focus on these benchmark fields:

```text
CPU_time_ms
GPU_basic_time_ms
GPU_tiled_time_ms
speedup_basic
speedup_tiled
basic_correctness
tiled_correctness
```

Say the performance result carefully:

```text
In the Colab Tesla T4 benchmark, the tiled CUDA implementation reached up to 213x speedup over the CPU reference for N=1024.
```

Do not present `213x` as a universal result. It is a measured result from one Colab Tesla T4 benchmark run.

### 6. Correctness

Explain that speed is only valid if the GPU output agrees with the CPU reference.

Validation points:

```text
Top-10 overlap: 10/10
Max absolute error: approximately 1e-6 to 2.5e-6
Correctness: PASS
```

The top-10 overlap is important because the dashboard depends on the ranking of highest-risk casualties.

### 7. Bottlenecks

Mention current bottlenecks:

* Global memory access in matrix multiplication
* Softmax row reductions
* Separate kernel launch overhead
* Host-to-device and device-to-host transfers
* Small `d_model = 64`, which limits how much work each matrix tile can reuse
* Shared-memory synchronization overhead

Future CUDA optimization directions:

* Kernel fusion
* Flash-Attention-style memory-efficient attention
* Warp-level Softmax reductions
* WMMA / Tensor Core matrix multiplication
* Streaming telemetry batches

### 8. Dashboard

Only after the CUDA explanation, open the React dashboard:

```powershell
cd frontend
npm run dev
```

Then open:

```text
http://localhost:5173
```

Show:

* Analytics for CPU vs CUDA benchmark visualization
* Tactical Command for casualty ranking and attention-derived prioritization
* System Architecture for the end-to-end data flow
* Mission Plan for the operational wrapper around the CUDA output

Explain that the dashboard visualizes CUDA-generated JSON outputs:

```text
frontend/public/data/benchmark_results.json
frontend/public/data/risk_ranking.json
frontend/public/data/attention_stats.json
```

### 9. YOLO / Human Review

Move to Computer Vision last.

Explain:

```text
YOLO and human-reviewed annotation are demo/integration layers.
CUDA attention is the core computational contribution.
```

The presentation Computer Vision page prioritizes:

```text
frontend/public/data/human_review_preview.jpg
frontend/public/data/human_review_detections.json
```

Raw YOLO debug output is hidden unless the URL contains:

```text
?debug=1
```

Use this final framing:

```text
Drone vision gives visual localization.
ResQBand telemetry gives medical severity.
CUDA attention ranks casualties in real time.
The dashboard turns it into operational decisions.
```

## Fallbacks

If local CUDA fails:

* Use the Colab benchmark and exported JSON files.

If the frontend does not start:

```powershell
cd frontend
npm install
npm run dev
```

If YOLO output is noisy:

* Use the human-reviewed demo preview.
* Keep raw YOLO only for debug mode.

## What Not To Claim

Do not claim that YOLO is the academic CUDA contribution.

Do not claim that the measured 213x speedup is guaranteed on every machine.

Do not claim the system is a clinical or operational medical decision system. It is a research and course demonstration of CUDA-accelerated attention applied to simulated battlefield triage.
