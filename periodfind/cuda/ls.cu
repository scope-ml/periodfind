// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#include "ls.h"

#include <algorithm>

#include <cstdio>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

const float TWO_PI = M_PI * 2.0;

//
// Simple LombScargle Function Definitions
//

LombScargle::LombScargle() {}

//
// CUDA Kernels
//

// Tile size for shared-memory light curve tiling
#define LS_TILE_SIZE 256

__global__ void LombScargleKernel(const float* __restrict__ times,
                                  const float* __restrict__ mags,
                                  const size_t length,
                                  const float* __restrict__ periods,
                                  const float* __restrict__ period_dts,
                                  const size_t num_periods,
                                  const size_t num_period_dts,
                                  const LombScargle params,
                                  float* __restrict__ periodogram) {
    // Shared memory for tiling light curve data
    extern __shared__ float sh_data[];
    float* sh_times = &sh_data[0];
    float* sh_mags = &sh_data[LS_TILE_SIZE];

    // One block per (period, period_dt) pair
    const size_t period_idx = blockIdx.x;
    const size_t pdt_idx = blockIdx.y;

    if (period_idx >= num_periods || pdt_idx >= num_period_dts) {
        return;
    }

    const float period = periods[period_idx];
    const float period_dt = period_dts[pdt_idx];
    const float pdt_corr = (period_dt / period) / 2;

    // Per-thread accumulators
    float mag_cos = 0.0f;
    float mag_sin = 0.0f;
    float cos_cos = 0.0f;
    float cos_sin = 0.0f;

    float cos_val, sin_val, i_part;

    // Process the light curve in tiles
    for (size_t tile_start = 0; tile_start < length; tile_start += LS_TILE_SIZE) {
        size_t tile_end = tile_start + LS_TILE_SIZE;
        if (tile_end > length) tile_end = length;
        size_t tile_len = tile_end - tile_start;

        // Cooperatively load tile into shared memory
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            sh_times[i] = times[tile_start + i];
            sh_mags[i] = mags[tile_start + i];
        }
        __syncthreads();

        // Each thread accumulates over the entire tile
        for (size_t i = 0; i < tile_len; i++) {
            float t = sh_times[i];
            float mag = sh_mags[i];

            float t_corr = t - pdt_corr * t * t;
            float folded = fabsf(modff(t_corr / period, &i_part));

            sincosf(TWO_PI * folded, &sin_val, &cos_val);

            mag_cos += mag * cos_val;
            mag_sin += mag * sin_val;
            cos_cos += cos_val * cos_val;
            cos_sin += cos_val * sin_val;
        }
        __syncthreads();
    }

    // Only thread 0 computes the final LS value (all threads have full sums)
    if (threadIdx.x != 0) return;

    float sin_sin = static_cast<float>(length) - cos_cos;

    float cos_tau, sin_tau;
    sincosf(0.5f * atan2f(2.0f * cos_sin, cos_cos - sin_sin), &sin_tau, &cos_tau);

    float numerator_l = cos_tau * mag_cos + sin_tau * mag_sin;
    numerator_l *= numerator_l;

    float numerator_r = cos_tau * mag_sin - sin_tau * mag_cos;
    numerator_r *= numerator_r;

    float denominator_l = cos_tau * cos_tau * cos_cos
                          + 2 * cos_tau * sin_tau * cos_sin
                          + sin_tau * sin_tau * sin_sin;

    float denominator_r = cos_tau * cos_tau * sin_sin
                          - 2 * cos_tau * sin_tau * cos_sin
                          + sin_tau * sin_tau * cos_cos;

    periodogram[period_idx * num_period_dts + pdt_idx] =
        0.5f * ((numerator_l / denominator_l) + (numerator_r / denominator_r));
}

//
// Wrapper Functions
//

float* LombScargle::DeviceCalcLS(const float* times,
                                 const float* mags,
                                 const size_t length,
                                 const float* periods,
                                 const float* period_dts,
                                 const size_t num_periods,
                                 const size_t num_p_dts) const {
    float* periodogram;
    gpuErrchk(
        cudaMalloc(&periodogram, num_periods * num_p_dts * sizeof(float)));

    // One block per (period, period_dt) pair
    const size_t num_threads = 256;
    const size_t shared_bytes = 2 * LS_TILE_SIZE * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    LombScargleKernel<<<grid_dim, num_threads, shared_bytes>>>(
        times, mags, length, periods, period_dts, num_periods, num_p_dts,
        *this, periodogram);

    return periodogram;
}

void LombScargle::CalcLS(float* times,
                         float* mags,
                         size_t length,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts,
                         float* per_out) const {
    CalcLSBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                  std::vector<size_t>{length}, periods, period_dts, num_periods,
                  num_p_dts, per_out);
}

float* LombScargle::CalcLS(float* times,
                           float* mags,
                           size_t length,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts) const {
    return CalcLSBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                         std::vector<size_t>{length}, periods, period_dts,
                         num_periods, num_p_dts);
}

void LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                const std::vector<float*>& mags,
                                const std::vector<size_t>& lengths,
                                const float* periods,
                                const float* period_dts,
                                const size_t num_periods,
                                const size_t num_p_dts,
                                float* per_out) const {
    // Size of one periodogram out array, and total periodogram output size.
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    size_t per_size_total = per_out_size * lengths.size();

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
    const size_t shared_bytes = 2 * LS_TILE_SIZE * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    // Buffer size (large enough for longest light curve)
    auto max_length = std::max_element(lengths.begin(), lengths.end());
    const size_t buffer_length = *max_length;
    const size_t buffer_bytes = sizeof(float) * buffer_length;

    // Create 2 CUDA streams for double-buffered async transfers
    const int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuErrchk(cudaStreamCreate(&streams[s]));
    }

    // Allocate double-buffered device memory
    float* dev_times_buf[NUM_STREAMS];
    float* dev_mags_buf[NUM_STREAMS];
    float* dev_per_buf[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuErrchk(cudaMalloc(&dev_times_buf[s], buffer_bytes));
        gpuErrchk(cudaMalloc(&dev_mags_buf[s], buffer_bytes));
        gpuErrchk(cudaMalloc(&dev_per_buf[s], per_out_size));
    }

    for (size_t i = 0; i < lengths.size(); i++) {
        int s = i % NUM_STREAMS;
        cudaStream_t stream = streams[s];

        // Copy light curve into device buffer (async)
        const size_t curve_bytes = lengths[i] * sizeof(float);
        gpuErrchk(cudaMemcpyAsync(dev_times_buf[s], times[i], curve_bytes,
                                   cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(dev_mags_buf[s], mags[i], curve_bytes,
                                   cudaMemcpyHostToDevice, stream));

        // Zero periodogram output
        gpuErrchk(cudaMemsetAsync(dev_per_buf[s], 0, per_out_size, stream));

        LombScargleKernel<<<grid_dim, num_threads, shared_bytes, stream>>>(
            dev_times_buf[s], dev_mags_buf[s], lengths[i], dev_periods,
            dev_period_dts, num_periods, num_p_dts, *this, dev_per_buf[s]);

        // Copy periodogram back to host (async)
        gpuErrchk(cudaMemcpyAsync(&per_out[i * per_points], dev_per_buf[s],
                                   per_out_size, cudaMemcpyDeviceToHost,
                                   stream));
    }

    // Synchronize and clean up streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuErrchk(cudaStreamSynchronize(streams[s]));
    }

    // Free all of the GPU memory
    gpuErrchk(cudaFree(dev_periods));
    gpuErrchk(cudaFree(dev_period_dts));
    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuErrchk(cudaFree(dev_times_buf[s]));
        gpuErrchk(cudaFree(dev_mags_buf[s]));
        gpuErrchk(cudaFree(dev_per_buf[s]));
        gpuErrchk(cudaStreamDestroy(streams[s]));
    }
}

float* LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                  const std::vector<float*>& mags,
                                  const std::vector<size_t>& lengths,
                                  const float* periods,
                                  const float* period_dts,
                                  const size_t num_periods,
                                  const size_t num_p_dts) const {
    // Size of one periodogram out array, and total periodogram output size.
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    size_t per_size_total = per_out_size * lengths.size();

    // Allocate the output CE array so we can copy to it.
    float* per_out = (float*)malloc(per_size_total);

    CalcLSBatched(times, mags, lengths, periods, period_dts, num_periods,
                  num_p_dts, per_out);

    return per_out;
}