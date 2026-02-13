# Benchmarks

## Methodology

All benchmarks use synthetic sinusoidal light curves with Gaussian noise
(amplitude 0.1, period 2.5, observation window 0--100).

- **CPU**: Rust/Rayon on 28-core Skylake Xeon
- **GPU**: 1x or 2x NVIDIA Tesla P100 (12 GB each)
- **Trial periods**: 1,000 linearly spaced between 0.5 and 10.0 (single `period_dt`)
- **Timing**: median of 3 runs after 1 warmup iteration
- **Metric**: throughput in total points processed per second (`n_curves * n_points / wall_sec`)

All backends (CPU, 1x P100, 2x P100) were benchmarked on the same compute
node to ensure a fair comparison.

## Point-Count Scaling

Fixed **1,000 curves**, varying points per curve from 64 to 16,384.

### Throughput table (points/sec)

| pts/curve | Backend | CE | AOV | LS | FPW | BLS |
|----------:|---------|---:|----:|---:|----:|----:|
| 256 | CPU | 498K | 581K | 527K | 658K | 469K |
| 256 | 1x P100 | 1.1M | 1.1M | 1.2M | 1.1M | 978K |
| 256 | 2x P100 | 1.2M | 1.3M | 1.3M | 1.3M | 1.2M |
| 1,024 | CPU | 938K | 1.1M | 979K | 1.2M | 1.1M |
| 1,024 | 1x P100 | 3.7M | 3.4M | 4.4M | 2.6M | 3.1M |
| 1,024 | 2x P100 | 4.4M | 4.3M | 5.1M | 3.7M | 4.0M |
| 4,096 | CPU | 1.2M | 1.4M | 1.3M | 1.9M | 1.8M |
| 4,096 | 1x P100 | 10.3M | 3.9M | 13.1M | 2.5M | 6.1M |
| 4,096 | 2x P100 | 13.9M | 6.8M | 16.5M | 4.5M | 9.8M |
| 16,384 | CPU | 1.3M | 1.5M | 1.4M | 2.1M | 2.1M |
| 16,384 | 1x P100 | 17.9M | 2.4M | 26.6M | 1.3M | 3.6M |
| 16,384 | 2x P100 | 28.9M | 4.8M | 40.5M | 2.6M | 7.0M |

Speedup is 1x P100 vs CPU.

### Throughput plot

![Point-count scaling](throughput_points.png)

Solid lines = 1x P100, dash-dot lines = 2x P100, dashed lines = CPU (Rust).

### Discussion

**CE and LS** are the strongest GPU beneficiaries. Lomb-Scargle reaches 41M
pts/sec on 2x P100 (29x over CPU at 16K points) because its per-point
trigonometric work maps efficiently to GPU SIMD lanes. Conditional Entropy
follows a similar pattern, reaching 29M pts/sec on 2x P100 (23x over CPU).

**AOV** sees diminishing GPU returns at large point counts. GPU throughput
peaks around 2K points then declines due to register/shared-memory pressure
limiting kernel occupancy. Even so, 2x P100 reaches 4.8M pts/sec at 16K
points (3.2x over CPU).

**FPW** GPU throughput peaks around 2K points (2.8M pts/sec on 1x P100)
then declines at larger sizes due to memory-access patterns in the
accumulation phase. The parallelized final-phase reduction ensures the GPU
stays faster than CPU up to 4K points on 1x P100 and up to 8K on 2x P100.

**BLS** benefits dramatically from the parallelized search kernel. Where the
old serial kernel was 5x slower than CPU, the new parallel kernel reaches
6.1M pts/sec on 1x P100 (3.4x over CPU at 4K points) and 9.8M on 2x P100.
At 16K points, BLS achieves 7.0M pts/sec on 2x P100 (3.4x over CPU).

**2x P100 scaling**: with threaded multi-GPU dispatch (one CPU thread per
GPU), the second GPU provides near-linear scaling at large point counts.
At 16K points per curve, 2x/1x ratios are: CE 1.6x, AOV 2.0x, LS 1.5x,
FPW 2.0x, BLS 1.9x.

## Curve-Count Scaling

Fixed **1,024 points/curve**, varying the number of curves: 100, 1,000, and 10,000.

### Throughput table (points/sec)

| curves | Backend | CE | AOV | LS | FPW | BLS |
|-------:|---------|---:|----:|---:|----:|----:|
| 100 | CPU | 937K | 1.1M | 972K | 1.4M | 1.1M |
| 100 | 1x P100 | 3.6M | 3.3M | 4.2M | 2.5M | 3.0M |
| 100 | 2x P100 | 4.3M | 4.2M | 4.8M | 3.5M | 3.9M |
| 1,000 | CPU | 943K | 1.1M | 978K | 1.2M | 1.0M |
| 1,000 | 1x P100 | 3.7M | 3.4M | 4.4M | 2.6M | 3.1M |
| 1,000 | 2x P100 | 4.5M | 4.4M | 5.1M | 3.7M | 4.2M |
| 10,000 | CPU | 938K | 1.1M | 977K | 1.2M | 1.0M |
| 10,000 | 1x P100 | 3.8M | 3.4M | 4.4M | 2.6M | 3.1M |
| 10,000 | 2x P100 | 4.5M | 4.4M | 5.1M | 3.7M | 4.2M |

### Throughput plot

![Curve-count scaling](throughput_curves.png)

### Discussion

GPU throughput is stable from 100 to 10,000 curves, indicating the GPU is
fully occupied even at 100 curves. CPU throughput is also flat thanks to
Rayon's work-stealing across 28 cores.

At 1,000 curves (the production-like configuration), 1x P100 is 2.1--4.5x
faster than CPU across all algorithms, and 2x P100 adds another 1.2--1.4x
on top.

BLS at 1024 pts/curve achieves 3.1M pts/sec on 1x P100 (3.0x over CPU)
and 4.2M on 2x P100 (4.0x over CPU), a dramatic improvement from the
pre-optimization state where the GPU was slower than CPU.

## Multi-Device Scaling

### CUDA multi-GPU

The GPU backend supports multi-GPU execution with one CPU thread per GPU
for concurrent device feeding. Curves are partitioned evenly across visible
devices using `cudaSetDevice`. Control which GPUs are used with the
`CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python my_script.py
```

At 1,000 curves with 1,024 points each, 2x P100 achieves 1.2--1.4x
throughput over 1x P100 across all algorithms. At larger point counts
(16K points), scaling approaches 1.5--2.0x, as GPU compute time dominates
over launch overhead.

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
# Run both point-scaling and curve-scaling sweeps (single GPU)
python benchmarks/throughput_bench.py

# Generate docs/throughput_points.png and docs/throughput_curves.png
python benchmarks/plot_throughput.py

# Multi-GPU benchmarks on a SLURM cluster with 2x P100
sbatch benchmarks/run_bench.sh
```

The benchmark writes results to `benchmarks/throughput_results.csv`. The
plotting script reads this CSV and produces two PNG files in the `docs/`
directory.
