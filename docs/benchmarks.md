# Benchmarks

## Methodology

All benchmarks use synthetic sinusoidal light curves with Gaussian noise
(amplitude 0.1, period 2.5, observation window 0–100).

- **CPU**: Rust/Rayon on 2x Intel Xeon E5-2680 v4 (28 cores total)
- **GPU**: single NVIDIA Tesla P100 (12 GB)
- **Trial periods**: 1,000 linearly spaced between 0.5 and 10.0 (single `period_dt`)
- **Timing**: median of 3 runs after 1 warmup iteration
- **Metric**: throughput in total points processed per second (`n_curves * n_points / wall_sec`)

## Point-Count Scaling

Fixed **100 curves**, varying points per curve from 64 to 16,384.

### Throughput table (points/sec)

| pts/curve | Backend | CE | AOV | LS | FPW | BLS |
|----------:|---------|---:|----:|---:|----:|----:|
| 256 | CPU | 99K | 116K | 95K | 113K | 56K |
| 256 | GPU | 930K | 956K | 1.0M | 720K | 97K |
| 256 | **Speedup** | **9.4x** | **8.2x** | **10.7x** | **6.4x** | **1.7x** |
| 1,024 | CPU | 136K | 137K | 126K | 140K | 112K |
| 1,024 | GPU | 3.2M | 3.0M | 3.8M | 2.4M | 378K |
| 1,024 | **Speedup** | **24x** | **22x** | **30x** | **17x** | **3.4x** |
| 4,096 | CPU | 145K | 132K | 142K | 162K | 193K |
| 4,096 | GPU | 8.9M | 3.5M | 11.8M | 2.4M | 1.4M |
| 4,096 | **Speedup** | **61x** | **26x** | **83x** | **15x** | **7.3x** |
| 16,384 | CPU | 133K | 150K | 117K | 181K | 194K |
| 16,384 | GPU | 16.9M | 2.3M | 25.1M | 1.3M | 2.2M |
| 16,384 | **Speedup** | **127x** | **16x** | **216x** | **7.2x** | **11x** |

### Throughput plot

![Point-count scaling](throughput_points.png)

Solid lines = GPU (CUDA), dashed lines = CPU (Rust).

### Discussion

The GPU advantage grows with curve length for CE, LS, and BLS. Lomb-Scargle
shows the largest speedup (up to 216x at 16K points) because its per-point
trigonometric work maps well to GPU SIMD lanes. AOV and FPW GPU throughput
plateaus or declines at large sizes (8K–16K points) due to reduced kernel
occupancy — these algorithms have higher per-point register/shared-memory
pressure that limits the number of concurrent threads.

CPU throughput is relatively flat across all algorithms (100K–200K pts/sec),
reflecting Rayon's work-stealing parallelism saturating the available cores.

## Curve-Count Scaling

Fixed **1,024 points/curve**, varying the number of curves from 1 to 512.

### Throughput table (points/sec)

| curves | Backend | CE | AOV | LS | FPW | BLS |
|-------:|---------|---:|----:|---:|----:|----:|
| 1 | CPU | 63K | 95K | 93K | 199K | 153K |
| 1 | GPU | 1.3M | 1.3M | 1.4M | 1.1M | 333K |
| 8 | CPU | 94K | 83K | 121K | 210K | 160K |
| 8 | GPU | 2.8M | 2.5M | 3.1M | 2.0M | 380K |
| 64 | CPU | 78K | 90K | 125K | 198K | 100K |
| 64 | GPU | 3.3M | 2.9M | 3.7M | 2.3M | 386K |
| 256 | CPU | 78K | 97K | 123K | 181K | 152K |
| 256 | GPU | 3.3M | 2.9M | 3.8M | 2.4M | 388K |
| 512 | CPU | 74K | 143K | 120K | 187K | 98K |
| 512 | GPU | 3.3M | 2.9M | 3.8M | 2.4M | 388K |

### Throughput plot

![Curve-count scaling](throughput_curves.png)

### Discussion

GPU throughput rises steeply from 1 to ~32 curves as the GPU fills its
streaming multiprocessors, then plateaus. At 512 curves the GPU processes
each curve as an independent work unit, achieving near-peak occupancy for all
algorithms.

CPU throughput is roughly constant regardless of batch size because Rayon
distributes curves across threads via `par_chunks_mut`. With 28 cores,
even small batches (4–8 curves) are enough to saturate the CPU.

BLS GPU throughput is lower overall due to the scan over transit-duration
fractions inside each period bin, which serializes more work per thread.

## Multi-Device Scaling

All benchmark numbers above were collected on a **single GPU**. The library
also supports multi-device configurations described below, but those were not
used in these measurements.

### CUDA multi-GPU

The GPU backend supports multi-GPU execution. Curves are partitioned evenly
across visible devices using `cudaSetDevice`. Control which GPUs are used with
the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python my_script.py
```

### Rust/Rayon CPU parallelism

The CPU backend uses Rayon's `par_chunks_mut` to distribute curves across
threads. The GIL is released during the Rust computation, so Python threads
are not blocked. Thread count follows Rayon's default (number of logical
cores) and can be overridden with `RAYON_NUM_THREADS`:

```bash
RAYON_NUM_THREADS=8 python my_script.py
```

## Reproducing

Run the benchmark suite and generate plots:

```bash
# Run both point-scaling and curve-scaling sweeps
python benchmarks/throughput_bench.py

# Generate docs/throughput_points.png and docs/throughput_curves.png
python benchmarks/plot_throughput.py
```

The benchmark writes results to `benchmarks/throughput_results.csv`. The
plotting script reads this CSV and produces two PNG files in the `docs/`
directory.
