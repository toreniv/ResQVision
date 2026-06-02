#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err__) << std::endl;     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

struct Soldier {
    int soldier_id;
    float x;
    float y;
    float heart_rate;
    float spo2;
    float temperature;
    float respiration_rate;
    float motion_level;
    float signal_quality;
    float battery_level;
};

struct BenchmarkRow {
    int N;
    float cpu_ms;
    float gpu_ms;
    float speedup;
    bool correctness;
    float max_abs_error;
    float mean_abs_error;
    int top10_overlap;
};

static float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

static std::vector<Soldier> generate_soldiers(int N) {
    std::mt19937 rng(1337 + N);
    std::uniform_real_distribution<float> pos_dist(0.0f, 1000.0f);
    std::uniform_real_distribution<float> hr_dist(62.0f, 118.0f);
    std::uniform_real_distribution<float> spo2_dist(93.0f, 99.0f);
    std::uniform_real_distribution<float> temp_dist(36.1f, 37.8f);
    std::uniform_real_distribution<float> resp_dist(12.0f, 24.0f);
    std::uniform_real_distribution<float> motion_dist(0.25f, 1.0f);
    std::uniform_real_distribution<float> quality_dist(0.72f, 1.0f);
    std::uniform_real_distribution<float> battery_dist(0.35f, 1.0f);

    std::vector<Soldier> soldiers(N);
    for (int i = 0; i < N; ++i) {
        soldiers[i] = {
            i,
            pos_dist(rng),
            pos_dist(rng),
            hr_dist(rng),
            spo2_dist(rng),
            temp_dist(rng),
            resp_dist(rng),
            motion_dist(rng),
            quality_dist(rng),
            battery_dist(rng)
        };
    }

    const int critical_count = std::min(8, N);
    for (int i = 0; i < critical_count; ++i) {
        int idx = (i * 53 + 17) % N;
        soldiers[idx].heart_rate = 138.0f + 6.0f * i;
        soldiers[idx].spo2 = 78.0f + static_cast<float>(i % 4);
        soldiers[idx].temperature = 38.5f + 0.15f * i;
        soldiers[idx].respiration_rate = (i % 2 == 0) ? 7.0f + i : 32.0f + i;
        soldiers[idx].motion_level = 0.02f + 0.015f * i;
        soldiers[idx].signal_quality = 0.55f;
        soldiers[idx].battery_level = 0.18f + 0.02f * i;
    }

    return soldiers;
}

static std::vector<float> build_feature_matrix(const std::vector<Soldier>& soldiers, int d_model) {
    const int N = static_cast<int>(soldiers.size());
    std::vector<float> X(N * d_model, 0.0f);

    for (int i = 0; i < N; ++i) {
        const Soldier& s = soldiers[i];
        float base[10] = {
            s.x / 1000.0f,
            s.y / 1000.0f,
            (s.heart_rate - 40.0f) / 160.0f,
            s.spo2 / 100.0f,
            (s.temperature - 34.0f) / 8.0f,
            s.respiration_rate / 45.0f,
            s.motion_level,
            s.signal_quality,
            s.battery_level,
            static_cast<float>(s.soldier_id % 97) / 97.0f
        };

        for (int j = 0; j < d_model; ++j) {
            float patterned = base[j % 10];
            float phase = 0.5f + 0.5f * std::sin(0.13f * static_cast<float>(i + 1) *
                                                static_cast<float>(j + 1));
            X[i * d_model + j] = 0.75f * patterned + 0.25f * phase;
        }
    }

    return X;
}

static std::vector<float> make_weight_matrix(int d_model, int seed_offset) {
    std::vector<float> W(d_model * d_model, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_model));
    for (int r = 0; r < d_model; ++r) {
        for (int c = 0; c < d_model; ++c) {
            float a = std::sin(0.017f * static_cast<float>((r + 1) * (c + 3 + seed_offset)));
            float b = std::cos(0.011f * static_cast<float>((r + 5 + seed_offset) * (c + 1)));
            W[r * d_model + c] = scale * (0.55f * a + 0.45f * b);
        }
    }
    return W;
}

static void matmul_cpu(const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C,
                       int rows_a,
                       int cols_a,
                       int cols_b) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < rows_a; ++i) {
        for (int k = 0; k < cols_a; ++k) {
            float a = A[i * cols_a + k];
            for (int j = 0; j < cols_b; ++j) {
                C[i * cols_b + j] += a * B[k * cols_b + j];
            }
        }
    }
}

static void qk_transpose_cpu(const std::vector<float>& Q,
                             const std::vector<float>& K,
                             std::vector<float>& scores,
                             int N,
                             int d_model) {
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_model));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; ++k) {
                sum += Q[i * d_model + k] * K[j * d_model + k];
            }
            scores[i * N + j] = sum * inv_sqrt_d;
        }
    }
}

static void softmax_rows_cpu(std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float row_max = matrix[i * cols];
        for (int j = 1; j < cols; ++j) {
            row_max = std::max(row_max, matrix[i * cols + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float e = std::exp(matrix[i * cols + j] - row_max);
            matrix[i * cols + j] = e;
            sum += e;
        }

        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] /= sum;
        }
    }
}

static void attention_cpu(const std::vector<float>& X,
                          const std::vector<float>& Wq,
                          const std::vector<float>& Wk,
                          const std::vector<float>& Wv,
                          std::vector<float>& output,
                          std::vector<float>* attention_out,
                          int N,
                          int d_model) {
    std::vector<float> Q(N * d_model), K(N * d_model), V(N * d_model);
    std::vector<float> scores(N * N), attention(N * N);

    matmul_cpu(X, Wq, Q, N, d_model, d_model);
    matmul_cpu(X, Wk, K, N, d_model, d_model);
    matmul_cpu(X, Wv, V, N, d_model, d_model);
    qk_transpose_cpu(Q, K, scores, N, d_model);
    attention = scores;
    softmax_rows_cpu(attention, N, N);
    matmul_cpu(attention, V, output, N, N, d_model);

    if (attention_out) {
        *attention_out = attention;
    }
}

// Thread mapping: blockIdx.y/threadIdx.y select a row and blockIdx.x/threadIdx.x
// select a column. Each valid thread computes one scores[row, col] element.
// The boundary check protects edge blocks when N is not a multiple of block size.
__global__ void qk_transpose_kernel(const float* Q,
                                    const float* K,
                                    float* scores,
                                    int N,
                                    int d_model) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d_model; ++k) {
            sum += Q[row * d_model + k] * K[col * d_model + k];
        }
        scores[row * N + col] = sum;
    }
}

// Same 2D mapping as QK^T. Each thread scales one scores matrix element.
__global__ void scale_kernel(float* scores, int N, float inv_sqrt_d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        scores[row * N + col] *= inv_sqrt_d;
    }
}

// One CUDA block owns one attention row. Threads in the block scan columns with
// a stride, then reduce max and sum in shared memory for stable row softmax.
__global__ void row_softmax_kernel(float* scores, int N) {
    extern __shared__ float shared[];
    float* max_cache = shared;
    float* sum_cache = shared + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, scores[row * N + col]);
    }
    max_cache[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_cache[tid] = fmaxf(max_cache[tid], max_cache[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = max_cache[0];
    float local_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float e = expf(scores[row * N + col] - row_max);
        scores[row * N + col] = e;
        local_sum += e;
    }
    sum_cache[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_cache[tid] += sum_cache[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = sum_cache[0];
    for (int col = tid; col < N; col += blockDim.x) {
        scores[row * N + col] /= row_sum;
    }
}

// Thread mapping: row is a soldier and col is an output embedding dimension.
// The basic version uses global memory only. Shared-memory tiling could be
// added later for QK^T and attention@V to reduce repeated global reads.
__global__ void attention_v_kernel(const float* attention,
                                   const float* V,
                                   float* output,
                                   int N,
                                   int d_model) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < d_model) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += attention[row * N + k] * V[k * d_model + col];
        }
        output[row * d_model + col] = sum;
    }
}

static void attention_gpu(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V,
                          std::vector<float>& output,
                          std::vector<float>* attention_out,
                          int N,
                          int d_model,
                          float* elapsed_ms) {
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_scores = nullptr, *d_output = nullptr;
    size_t nd_bytes = static_cast<size_t>(N) * d_model * sizeof(float);
    size_t nn_bytes = static_cast<size_t>(N) * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q, nd_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, nd_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, nd_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, nn_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, nd_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), nd_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K.data(), nd_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V.data(), nd_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    dim3 block2d(16, 16);
    dim3 grid_scores((N + block2d.x - 1) / block2d.x,
                     (N + block2d.y - 1) / block2d.y);
    qk_transpose_kernel<<<grid_scores, block2d>>>(d_Q, d_K, d_scores, N, d_model);
    CUDA_CHECK(cudaGetLastError());

    scale_kernel<<<grid_scores, block2d>>>(d_scores, N, 1.0f / std::sqrt(static_cast<float>(d_model)));
    CUDA_CHECK(cudaGetLastError());

    int softmax_threads = 256;
    size_t shared_bytes = static_cast<size_t>(softmax_threads) * 2 * sizeof(float);
    row_softmax_kernel<<<N, softmax_threads, shared_bytes>>>(d_scores, N);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_output((d_model + block2d.x - 1) / block2d.x,
                     (N + block2d.y - 1) / block2d.y);
    attention_v_kernel<<<grid_output, block2d>>>(d_scores, d_V, d_output, N, d_model);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, nd_bytes, cudaMemcpyDeviceToHost));
    if (attention_out) {
        attention_out->resize(static_cast<size_t>(N) * N);
        CUDA_CHECK(cudaMemcpy(attention_out->data(), d_scores, nn_bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_output));
}

static float compute_risk_score(const Soldier& s, const std::vector<float>& output, int row, int d_model) {
    float mean_abs = 0.0f;
    float max_abs = 0.0f;
    for (int j = 0; j < d_model; ++j) {
        float value = std::fabs(output[row * d_model + j]);
        mean_abs += value;
        max_abs = std::max(max_abs, value);
    }
    mean_abs /= static_cast<float>(d_model);

    float spo2_risk = clamp01((94.0f - s.spo2) / 18.0f);
    float hr_risk = clamp01((s.heart_rate - 110.0f) / 70.0f);
    float resp_low = clamp01((10.0f - s.respiration_rate) / 8.0f);
    float resp_high = clamp01((s.respiration_rate - 24.0f) / 18.0f);
    float motion_risk = clamp01((0.35f - s.motion_level) / 0.35f);
    float attention_risk = clamp01(0.25f * mean_abs + 0.10f * max_abs);

    return 100.0f * clamp01(0.34f * spo2_risk +
                            0.22f * hr_risk +
                            0.18f * std::max(resp_low, resp_high) +
                            0.16f * motion_risk +
                            0.10f * attention_risk);
}

static std::string risk_category(float score) {
    if (score >= 65.0f) return "critical";
    if (score >= 35.0f) return "urgent";
    return "stable";
}

static std::vector<float> compute_risks(const std::vector<Soldier>& soldiers,
                                        const std::vector<float>& output,
                                        int d_model) {
    std::vector<float> risks(soldiers.size(), 0.0f);
    for (int i = 0; i < static_cast<int>(soldiers.size()); ++i) {
        risks[i] = compute_risk_score(soldiers[i], output, i, d_model);
    }
    return risks;
}

static std::vector<int> ranking_indices(const std::vector<float>& risks) {
    std::vector<int> idx(risks.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (risks[a] == risks[b]) return a < b;
        return risks[a] > risks[b];
    });
    return idx;
}

static int top10_overlap(const std::vector<int>& a, const std::vector<int>& b) {
    int limit = std::min(10, static_cast<int>(std::min(a.size(), b.size())));
    int overlap = 0;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < limit; ++j) {
            if (a[i] == b[j]) {
                ++overlap;
                break;
            }
        }
    }
    return overlap;
}

static void error_metrics(const std::vector<float>& cpu,
                          const std::vector<float>& gpu,
                          float* max_abs_error,
                          float* mean_abs_error) {
    float max_err = 0.0f;
    double sum_err = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        float err = std::fabs(cpu[i] - gpu[i]);
        max_err = std::max(max_err, err);
        sum_err += static_cast<double>(err);
    }
    *max_abs_error = max_err;
    *mean_abs_error = static_cast<float>(sum_err / static_cast<double>(cpu.size()));
}

static float median_ms(std::vector<float> times) {
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static BenchmarkRow run_case(int N, int d_model, bool keep_default_outputs,
                             std::vector<Soldier>* default_soldiers,
                             std::vector<float>* default_gpu_output,
                             std::vector<float>* default_gpu_attention,
                             std::vector<float>* default_gpu_risks) {
    std::vector<Soldier> soldiers = generate_soldiers(N);
    std::vector<float> X = build_feature_matrix(soldiers, d_model);
    std::vector<float> Wq = make_weight_matrix(d_model, 1);
    std::vector<float> Wk = make_weight_matrix(d_model, 17);
    std::vector<float> Wv = make_weight_matrix(d_model, 31);
    std::vector<float> Q(N * d_model), K(N * d_model), V(N * d_model);
    std::vector<float> cpu_output(N * d_model), gpu_output(N * d_model);
    std::vector<float> gpu_attention;

    matmul_cpu(X, Wq, Q, N, d_model, d_model);
    matmul_cpu(X, Wk, K, N, d_model, d_model);
    matmul_cpu(X, Wv, V, N, d_model, d_model);

    const int benchmark_repeats = 5;
    std::vector<float> cpu_times;
    std::vector<float> gpu_times;
    cpu_times.reserve(benchmark_repeats);
    gpu_times.reserve(benchmark_repeats);

    for (int repeat = 0; repeat < benchmark_repeats; ++repeat) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        attention_cpu(X, Wq, Wk, Wv, cpu_output, nullptr, N, d_model);
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        cpu_times.push_back(std::chrono::duration<float, std::milli>(cpu_stop - cpu_start).count());
    }

    std::vector<float> warmup_output(N * d_model);
    float warmup_ms = 0.0f;
    attention_gpu(Q, K, V, warmup_output, nullptr, N, d_model, &warmup_ms);

    for (int repeat = 0; repeat < benchmark_repeats; ++repeat) {
        float elapsed_ms = 0.0f;
        bool capture_outputs = keep_default_outputs && (repeat == benchmark_repeats - 1);
        attention_gpu(Q, K, V, gpu_output, capture_outputs ? &gpu_attention : nullptr,
                      N, d_model, &elapsed_ms);
        gpu_times.push_back(elapsed_ms);
    }

    float cpu_ms = median_ms(cpu_times);
    float gpu_ms = median_ms(gpu_times);

    float max_err = 0.0f;
    float mean_err = 0.0f;
    error_metrics(cpu_output, gpu_output, &max_err, &mean_err);
    bool pass = max_err < 1e-3f;

    std::vector<float> cpu_risks = compute_risks(soldiers, cpu_output, d_model);
    std::vector<float> gpu_risks = compute_risks(soldiers, gpu_output, d_model);
    int overlap = top10_overlap(ranking_indices(cpu_risks), ranking_indices(gpu_risks));

    if (keep_default_outputs) {
        *default_soldiers = soldiers;
        *default_gpu_output = gpu_output;
        *default_gpu_attention = gpu_attention;
        *default_gpu_risks = gpu_risks;
    }

    return {N, cpu_ms, gpu_ms, cpu_ms / gpu_ms, pass, max_err, mean_err, overlap};
}

static void write_benchmark_csv(const std::vector<BenchmarkRow>& rows) {
    std::ofstream out("benchmark_results.csv");
    out << "N,CPU_time_ms,GPU_time_ms,speedup,correctness,max_abs_error,mean_abs_error,top10_overlap\n";
    out << std::fixed << std::setprecision(6);
    for (const BenchmarkRow& row : rows) {
        out << row.N << ","
            << row.cpu_ms << ","
            << row.gpu_ms << ","
            << row.speedup << ","
            << (row.correctness ? "PASS" : "FAIL") << ","
            << row.max_abs_error << ","
            << row.mean_abs_error << ","
            << row.top10_overlap << "\n";
    }
}

static void write_risk_csv(const std::vector<Soldier>& soldiers,
                           const std::vector<float>& risks) {
    std::vector<int> ranking = ranking_indices(risks);
    std::ofstream out("risk_ranking.csv");
    out << "soldier_id,x_position,y_position,risk_score,heart_rate,spo2,respiration,motion_level,category\n";
    out << std::fixed << std::setprecision(4);
    for (int idx : ranking) {
        const Soldier& s = soldiers[idx];
        out << s.soldier_id << ","
            << s.x << ","
            << s.y << ","
            << risks[idx] << ","
            << s.heart_rate << ","
            << s.spo2 << ","
            << s.respiration_rate << ","
            << s.motion_level << ","
            << risk_category(risks[idx]) << "\n";
    }
}

static void write_attention_stats_csv(const std::vector<float>& attention, int N) {
    std::ofstream out("attention_stats.csv");
    out << "soldier_id,max_attention,mean_attention,entropy\n";
    out << std::fixed << std::setprecision(8);
    for (int i = 0; i < N; ++i) {
        float max_attention = 0.0f;
        float entropy = 0.0f;
        for (int j = 0; j < N; ++j) {
            float value = attention[i * N + j];
            max_attention = std::max(max_attention, value);
            if (value > 0.0f) {
                entropy -= value * std::log(value);
            }
        }
        out << i << ","
            << max_attention << ","
            << (1.0f / static_cast<float>(N)) << ","
            << entropy << "\n";
    }
}

static void write_attention_heatmap_csv(const std::vector<float>& attention, int N) {
    std::ofstream out("attention_heatmap.csv");
    out << "source_soldier,target_soldier,attention_weight\n";
    out << std::fixed << std::setprecision(8);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out << i << "," << j << "," << attention[i * N + j] << "\n";
        }
    }
}

int main() {
    const int d_model = 64;
    const int default_N = 512;
    std::vector<int> sizes = {128, 256, 512, 1024};

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA-capable GPU detected." << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "ResQVision CUDA attention core\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "d_model: " << d_model << "\n\n";

    std::vector<BenchmarkRow> rows;
    std::vector<Soldier> default_soldiers;
    std::vector<float> default_gpu_output;
    std::vector<float> default_gpu_attention;
    std::vector<float> default_gpu_risks;

    std::cout << "N | CPU time ms | GPU time ms | speedup | correctness\n";
    std::cout << "--|-------------|-------------|---------|------------\n";
    for (int N : sizes) {
        bool keep_default = (N == default_N);
        BenchmarkRow row = run_case(N, d_model, keep_default,
                                    &default_soldiers,
                                    &default_gpu_output,
                                    &default_gpu_attention,
                                    &default_gpu_risks);
        rows.push_back(row);
        std::cout << row.N << " | "
                  << std::fixed << std::setprecision(3)
                  << row.cpu_ms << " | "
                  << row.gpu_ms << " | "
                  << row.speedup << " | "
                  << (row.correctness ? "PASS" : "FAIL") << "\n";
        std::cout << "Max abs error: " << std::scientific << row.max_abs_error << "\n";
        std::cout << "Mean abs error: " << std::scientific << row.mean_abs_error << "\n";
        std::cout << "Correctness: " << (row.correctness ? "PASS" : "FAIL") << "\n";
        std::cout << "Top-10 overlap: " << row.top10_overlap << "/10\n\n";
    }

    write_benchmark_csv(rows);
    write_risk_csv(default_soldiers, default_gpu_risks);
    write_attention_stats_csv(default_gpu_attention, default_N);
    write_attention_heatmap_csv(default_gpu_attention, default_N);

    std::cout << "Wrote benchmark_results.csv\n";
    std::cout << "Wrote risk_ranking.csv\n";
    std::cout << "Wrote attention_stats.csv\n";
    std::cout << "Wrote attention_heatmap.csv\n";

    return EXIT_SUCCESS;
}
