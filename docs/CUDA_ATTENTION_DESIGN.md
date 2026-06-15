# CUDA Attention Design Notes

This document explains the CUDA part of ResQVision for the GPU Parallel Hardware Accelerator mini project.

## Required Attention Formula

The implemented computation is:

```text
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d)) * V
```

For the benchmark configuration:

```text
N = number of simulated soldiers

d = d_model = 64
```

The CUDA benchmark tests `N = 128, 256, 512, 1024`.

## Kernel Decomposition

The attention pipeline is decomposed into separate CUDA kernels:

1. `qk_transpose_kernel` - basic `Q * K^T`.
2. `qk_transpose_tiled_kernel` - tiled shared-memory `Q * K^T`.
3. `scale_kernel` - divide every score by `sqrt(d)`.
4. `row_softmax_kernel` - row-wise Softmax using shared-memory reductions.
5. `attention_v_kernel` - basic `softmax(scores) * V`.
6. `attention_v_tiled_kernel` - tiled shared-memory `softmax(scores) * V`.

The basic path keeps separate kernels for clarity and course compliance. The tiled path demonstrates the improved shared-memory implementation.

## Thread Mapping

For the matrix kernels, each CUDA thread computes one output element.

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

For `QK^T`:

```text
scores[row][col] = dot(Q[row], K[col])
```

For `Attention x V`:

```text
output[row][col] = dot(attention[row], V[:, col])
```

This is a direct mapping from a 2D matrix to a 2D CUDA grid. Boundary checks prevent illegal access when the matrix size is not exactly divisible by the block size.

## Block and Tile Size

The implementation uses:

```text
TILE_WIDTH = 16
blockDim = 16 x 16 = 256 threads
```

This was chosen because:

- 16x16 maps naturally to matrix tiles.
- 256 threads per block is a common practical CUDA size.
- It remains below the 1024-thread block limit.
- Shared memory per block stays small:

```text
2 x 16 x 16 x sizeof(float) = 2048 bytes
```

## Shared Memory Tiling

The tiled kernels reduce repeated global-memory reads.

### Tiled QK^T

Each block computes a `16 x 16` tile of the `N x N` score matrix.

For every tile step along `d_model`:

```text
1. Load a tile of Q into shared memory.
2. Load a tile of K into shared memory.
3. Synchronize all threads in the block.
4. Accumulate partial dot products using shared memory.
5. Synchronize again before loading the next tile.
```

### Tiled Attention x V

Each block computes a `16 x 16` tile of the output matrix.

For every tile step along `N`:

```text
1. Load a tile of the attention matrix.
2. Load a tile of V.
3. Synchronize.
4. Accumulate partial dot products.
5. Synchronize before the next tile.
```

## Softmax Reduction

Softmax is computed per row. One CUDA block owns one row.

The block performs two reductions in shared memory:

1. Row maximum reduction.
2. Row exponential sum reduction.

The row maximum is subtracted before exponentiation for numerical stability:

```text
exp(x_i - max(row))
```

## Benchmark Methodology

The code compares:

1. Naive CPU attention core.
2. Basic CUDA attention core.
3. Tiled CUDA attention core.

Q, K, and V are generated once before timing. The timed region compares only:

```text
QK^T -> Scaling -> Softmax -> Attention x V
```

This avoids unfairly charging Q/K/V projection to only the CPU path.

CUDA timing uses CUDA events around the kernel sequence. Host-to-device copies and allocation are outside the timed kernel region.

## Correctness Checks

Correctness is validated by comparing CPU output to GPU output:

- Maximum absolute error.
- Mean absolute error.
- Top-10 evacuation ranking overlap.

The CSV includes both basic and tiled validation columns.

## Output Files

Running `resqvision.cu` generates:

```text
benchmark_results.csv
risk_ranking.csv
attention_stats.csv
attention_heatmap.csv
```

`benchmark_results.csv` includes:

```text
CPU_time_ms
GPU_time_ms
GPU_basic_time_ms
GPU_tiled_time_ms
speedup
speedup_basic
speedup_tiled
correctness
basic_correctness
tiled_correctness
```

`GPU_time_ms`, `speedup`, and `correctness` point to the tiled result to keep the existing dashboard schema stable.

## Bottlenecks

Remaining bottlenecks:

1. Softmax requires multiple passes over every row.
2. Separate kernels add launch overhead.
3. FP32 tiled kernels do not use Tensor Cores.
4. Small matrix sizes may not benefit strongly from tiling.
5. A real streaming system must still account for host/device transfer and telemetry ingestion overhead.

## Future Work

Further CUDA optimizations could include:

- Fusing QK^T with scaling.
- Flash-Attention-style memory-efficient attention.
- Warp-level Softmax reductions.
- WMMA/Tensor Core matrix multiplication.
- Streaming batches of battlefield telemetry.
