# ResQVision Demo Steps

## Presentation Order

### 1. CUDA Problem and Attention Formula

Start with the core computational problem:

```text
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d)) * V
```

Explain that `N` is the number of simulated soldiers and `d = 64` is the attention feature dimension. The goal is to rank casualties by processing battlefield telemetry in parallel.

### 2. CUDA Kernels

Show that the CUDA implementation decomposes attention into kernels:

1. `Q * K^T`
2. Scaling by `1 / sqrt(d)`
3. Row-wise Softmax
4. `Attention * V`

Then explain that the project compares:

1. CPU reference
2. CUDA basic
3. CUDA tiled

The tiled CUDA path uses shared memory for both `Q * K^T` and `Attention * V`.

### 3. Thread and Block Mapping

Show the matrix thread mapping:

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

Each thread computes one output matrix element.

The tiled implementation uses:

```text
TILE_WIDTH = 16
16 x 16 threads per block
```

### 4. CPU vs CUDA Basic vs CUDA Tiled Benchmark

Present the Colab Tesla T4 benchmark:

| N | CPU Reference | CUDA Basic | CUDA Tiled | Tiled Speedup vs CPU |
|---:|---:|---:|---:|---:|
| 128 | 2.674 ms | 0.088 ms | 0.041 ms | 64.819x |
| 256 | 11.259 ms | 0.243 ms | 0.086 ms | 131.187x |
| 512 | 46.049 ms | 0.836 ms | 0.248 ms | 185.682x |
| 1024 | 183.423 ms | 3.200 ms | 0.861 ms | 213.061x |

Say this carefully:

```text
In the Colab Tesla T4 benchmark, the tiled CUDA implementation reached up to 213x speedup over the CPU reference for N=1024.
```

Do not present `213x` as a universal result.

### 5. Correctness Validation

Explain that speed is not enough; the GPU result is validated against the CPU reference.

Validation results:

```text
Top-10 overlap: 10/10
Max absolute error: approximately 1e-6 to 2.5e-6
Correctness: PASS
```

The top-10 overlap matters because the dashboard depends on the ranking of highest-risk casualties.

### 6. Bottlenecks and Future Optimizations

Mention current bottlenecks:

* Global memory access
* Softmax row reductions
* Separate kernel launch overhead
* Host-device transfers
* FP32 kernels not using Tensor Cores

Future CUDA optimizations:

* Kernel fusion
* Flash-Attention-style memory-efficient attention
* Warp-level Softmax
* WMMA / Tensor Core matrix multiplication
* Streaming telemetry batches

### 7. React Dashboard

Open the dashboard:

```powershell
cd frontend
npm run dev
```

Then open:

```text
http://localhost:5173
```

Show:

* Mission Plan
* Tactical Command
* Analytics
* System Architecture

Explain that the dashboard visualizes CUDA-generated JSON outputs:

```text
frontend/public/data/benchmark_results.json
frontend/public/data/risk_ranking.json
frontend/public/data/attention_stats.json
```

### 8. YOLO / Human Review Visual Demo Layer

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

### 9. Final Project Message

Use this closing message:

```text
Drone vision gives visual localization.
ResQBand telemetry gives medical severity.
CUDA attention ranks casualties in real time.
The dashboard turns it into operational decisions.
```

## Fallbacks

If local CUDA fails:

* Use the Colab benchmark and exported JSON files.

If frontend startup fails:

```powershell
cd frontend
npm install
npm run dev
```

If YOLO output is noisy:

* Use the human-reviewed demo preview.
* Keep raw YOLO only for debug mode.
