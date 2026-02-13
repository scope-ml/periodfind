// Fast Phase-folding Weighted (FPW) algorithm — CUDA implementation.
//
// Implements the FPW statistic from Finkbeiner et al. 2025.

#include "fpw.h"

#include <algorithm>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

//
// Simple FPW Function Definitions
//

FPW::FPW(size_t n_bins) {
    num_bins = n_bins;
    bin_size = 1.0 / static_cast<float>(n_bins);
}

__host__ __device__ size_t FPW::NumBins() const {
    return num_bins;
}

__host__ __device__ size_t FPW::PhaseBin(float phase_val) const {
    return static_cast<size_t>(phase_val / bin_size);
}

//
// CUDA Kernels
//

// Tile size for shared-memory light curve tiling
#define FPW_TILE_SIZE 256

__global__ void FPWKernel(const float* __restrict__ times,
                          const float* __restrict__ mags,
                          const float* __restrict__ ivar,
                          const float* __restrict__ ivar_y,
                          const size_t length,
                          const float* __restrict__ periods,
                          const float* __restrict__ period_dts,
                          const size_t num_periods,
                          const size_t num_period_dts,
                          const FPW params,
                          float* __restrict__ fpw_out) {
    // Shared memory layout:
    // [0 .. FPW_TILE_SIZE-1]                       = sh_times
    // [FPW_TILE_SIZE .. 2*FPW_TILE_SIZE-1]         = sh_ivar
    // [2*FPW_TILE_SIZE .. 3*FPW_TILE_SIZE-1]       = sh_ivar_y
    // [3*FPW_TILE_SIZE .. 3*FPW_TILE_SIZE+n_bins-1] = sh_vtcinvv
    // [3*FPW_TILE_SIZE+n_bins .. ]                  = sh_ytcinvv
    extern __shared__ float sh_data[];
    float* sh_times = &sh_data[0];
    float* sh_ivar = &sh_data[FPW_TILE_SIZE];
    float* sh_ivar_y = &sh_data[2 * FPW_TILE_SIZE];

    // One block per (period, period_dt) pair
    const size_t period_idx = blockIdx.x;
    const size_t pdt_idx = blockIdx.y;

    if (period_idx >= num_periods || pdt_idx >= num_period_dts) {
        return;
    }

    const float period = periods[period_idx];
    const float period_dt = period_dts[pdt_idx];
    const float pdt_corr = (period_dt / period) / 2;

    const size_t n_bins = params.NumBins();

    // Shared memory bin accumulators (after tile data)
    float* sh_vtcinvv = &sh_data[3 * FPW_TILE_SIZE];
    float* sh_ytcinvv = &sh_data[3 * FPW_TILE_SIZE + n_bins];

    // Cooperatively zero shared memory bins
    for (size_t k = threadIdx.x; k < n_bins; k += blockDim.x) {
        sh_vtcinvv[k] = 0.0f;
        sh_ytcinvv[k] = 0.0f;
    }
    __syncthreads();

    float i_part;

    // Process the light curve in tiles
    for (size_t tile_start = 0; tile_start < length;
         tile_start += FPW_TILE_SIZE) {
        size_t tile_end = tile_start + FPW_TILE_SIZE;
        if (tile_end > length)
            tile_end = length;
        size_t tile_len = tile_end - tile_start;

        // Cooperatively load tile into shared memory
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            sh_times[i] = times[tile_start + i];
            sh_ivar[i] = ivar[tile_start + i];
            sh_ivar_y[i] = ivar_y[tile_start + i];
        }
        __syncthreads();

        // All threads accumulate over their stripe of the tile
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            float t = sh_times[i];
            float t_corr = t - pdt_corr * t * t;
            float folded = fabsf(modff(t_corr / period, &i_part));

            size_t bin = params.PhaseBin(folded);
            if (bin >= n_bins)
                bin = n_bins - 1;

            atomicAdd(&sh_vtcinvv[bin], sh_ivar[i]);
            atomicAdd(&sh_ytcinvv[bin], sh_ivar_y[i]);
        }
        __syncthreads();
    }

    // Thread 0 computes the final FPW statistic from shared bins
    if (threadIdx.x != 0)
        return;

    float delta_chi = 0.0f;
    for (size_t k = 0; k < n_bins; k++) {
        if (sh_vtcinvv[k] > 0.0f) {
            delta_chi += sh_ytcinvv[k] * sh_ytcinvv[k] / (2.0f * sh_vtcinvv[k]);
        }
    }

    fpw_out[period_idx * num_period_dts + pdt_idx] = delta_chi;
}

//
// Wrapper Functions
//

void FPW::CalcFPWBatched(const std::vector<float*>& times,
                         const std::vector<float*>& mags,
                         const std::vector<float*>& errs,
                         const std::vector<size_t>& lengths,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts,
                         float* fpw_out) const {
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);

    // Copy trial information over
    float* dev_periods;
    float* dev_period_dts;
    gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));
    gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Kernel launch information: one block per (period, period_dt) pair
    const size_t num_threads = 256;
    const size_t shared_bytes =
        (3 * FPW_TILE_SIZE + 2 * NumBins()) * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    // Buffer size (large enough for longest light curve)
    auto max_length = std::max_element(lengths.begin(), lengths.end());
    const size_t buffer_length = *max_length;
    const size_t buffer_bytes = sizeof(float) * buffer_length;

    // Create 2 CUDA streams for double-buffered async transfers
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuErrchk(cudaStreamCreate(&streams[s]));
    }

    // Allocate double-buffered device memory
    float* dev_times_buf[NUM_STREAMS];
    float* dev_ivar_buf[NUM_STREAMS];
    float* dev_ivar_y_buf[NUM_STREAMS];
    float* dev_fpw_buf[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuErrchk(cudaMalloc(&dev_times_buf[s], buffer_bytes));
        gpuErrchk(cudaMalloc(&dev_ivar_buf[s], buffer_bytes));
        gpuErrchk(cudaMalloc(&dev_ivar_y_buf[s], buffer_bytes));
        gpuErrchk(cudaMalloc(&dev_fpw_buf[s], per_out_size));
    }

    // Host-side buffers for precomputed ivar and ivar_y
    float* h_ivar = (float*)malloc(buffer_bytes);
    float* h_ivar_y = (float*)malloc(buffer_bytes);

    for (size_t i = 0; i < lengths.size(); i++) {
        int s = i % NUM_STREAMS;
        cudaStream_t stream = streams[s];

        // Precompute inverse variance and weighted data on CPU
        for (size_t j = 0; j < lengths[i]; j++) {
            float e = errs[i][j];
            h_ivar[j] = 1.0f / (e * e);
            h_ivar_y[j] = h_ivar[j] * mags[i][j];
        }

        // Copy light curve data into device buffers (async)
        const size_t curve_bytes = lengths[i] * sizeof(float);
        gpuErrchk(cudaMemcpyAsync(dev_times_buf[s], times[i], curve_bytes,
                                  cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(dev_ivar_buf[s], h_ivar, curve_bytes,
                                  cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(dev_ivar_y_buf[s], h_ivar_y, curve_bytes,
                                  cudaMemcpyHostToDevice, stream));

        // Zero output
        gpuErrchk(cudaMemsetAsync(dev_fpw_buf[s], 0, per_out_size, stream));

        FPWKernel<<<grid_dim, num_threads, shared_bytes, stream>>>(
            dev_times_buf[s], NULL, dev_ivar_buf[s], dev_ivar_y_buf[s],
            lengths[i], dev_periods, dev_period_dts, num_periods, num_p_dts,
            *this, dev_fpw_buf[s]);

        // Copy result back to host (async)
        gpuErrchk(cudaMemcpyAsync(&fpw_out[i * per_points], dev_fpw_buf[s],
                                  per_out_size, cudaMemcpyDeviceToHost,
                                  stream));
    }

    // Synchronize and clean up streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuErrchk(cudaStreamSynchronize(streams[s]));
    }

    // Free all GPU memory
    gpuErrchk(cudaFree(dev_periods));
    gpuErrchk(cudaFree(dev_period_dts));
    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuErrchk(cudaFree(dev_times_buf[s]));
        gpuErrchk(cudaFree(dev_ivar_buf[s]));
        gpuErrchk(cudaFree(dev_ivar_y_buf[s]));
        gpuErrchk(cudaFree(dev_fpw_buf[s]));
        gpuErrchk(cudaStreamDestroy(streams[s]));
    }

    free(h_ivar);
    free(h_ivar_y);
}

float* FPW::CalcFPWBatched(const std::vector<float*>& times,
                           const std::vector<float*>& mags,
                           const std::vector<float*>& errs,
                           const std::vector<size_t>& lengths,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts) const {
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    size_t per_size_total = per_out_size * lengths.size();

    float* fpw_out = (float*)malloc(per_size_total);

    CalcFPWBatched(times, mags, errs, lengths, periods, period_dts, num_periods,
                   num_p_dts, fpw_out);

    return fpw_out;
}
