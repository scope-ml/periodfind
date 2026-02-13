# PeriodFind

A collection of CUDA-accelerated periodicity detection algorithms, with both C++ and Python APIs. Includes a Rust-based CPU backend for environments without GPU hardware.

## Algorithms

### Period-Finding

| Algorithm | Unified API | GPU (CUDA) | CPU (Rust) |
|-----------|-------------|-----------|------------|
| Conditional Entropy | `periodfind.ConditionalEntropy` | `periodfind.gpu.ConditionalEntropy` | `periodfind.cpu.ConditionalEntropy` |
| Analysis of Variance | `periodfind.AOV` | `periodfind.gpu.AOV` | `periodfind.cpu.AOV` |
| Lomb-Scargle | `periodfind.LombScargle` | `periodfind.gpu.LombScargle` | `periodfind.cpu.LombScargle` |
| Fast Phase-folding Weighted | `periodfind.FPW` | `periodfind.gpu.FPW` | `periodfind.cpu.FPW` |
| Box Least Squares | `periodfind.BoxLeastSquares` | `periodfind.gpu.BoxLeastSquares` | `periodfind.cpu.BoxLeastSquares` |

### Feature Extraction

| Algorithm | Unified API | CPU (Rust) |
|-----------|-------------|------------|
| Fourier Decomposition | `periodfind.FourierDecomposition` | `periodfind.cpu.FourierDecomposition` |

Fourier decomposition computes weighted linear least-squares Fourier fits with BIC model selection (0-5 harmonics) for a batch of light curves given pre-determined periods. Returns 14 features per curve: `[power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]`. This replaces the per-source `scipy.optimize.curve_fit` approach with a direct Cholesky solve, giving identical results orders of magnitude faster.

## Device API

Periodfind provides a PyTorch-style device abstraction so you can write device-agnostic code. When no device is set, it auto-detects GPU availability (tries to import the CUDA extensions and runs `nvidia-smi`).

```python
import periodfind

# Set the global default device
periodfind.set_device('cpu')   # or 'gpu'
print(periodfind.get_device()) # 'cpu'

# Factory functions dispatch to the right backend
ce  = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)
aov = periodfind.AOV(n_phase=15)
ls  = periodfind.LombScargle()
fpw = periodfind.FPW(n_bins=10)
bls = periodfind.BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.5)
fd  = periodfind.FourierDecomposition()  # CPU-only for now

# Per-call override (ignores the global default)
ce_gpu = periodfind.ConditionalEntropy(n_phase=10, n_mag=10, device='gpu')
```

You can still import backends directly:

```python
from periodfind.gpu import ConditionalEntropy  # CUDA backend
from periodfind.cpu import ConditionalEntropy  # Rust CPU backend
from periodfind.cpu import FourierDecomposition  # Rust CPU only
```

### Box Least Squares Usage

BLS searches for periodic box-shaped (flat-bottom) transit dips in time-series data ([Kovacs, Zucker & Mazeh 2002](https://ui.adsabs.harvard.edu/abs/2002A%26A...391..369K)). It is particularly well-suited for detecting eclipsing binaries and transiting exoplanets.

```python
import numpy as np
import periodfind

bls = periodfind.BoxLeastSquares(
    n_bins=50,     # number of phase bins
    qmin=0.01,     # minimum transit duration (fraction of period)
    qmax=0.5,      # maximum transit duration (fraction of period)
)

# times, mags: lists of float32 arrays (one per light curve)
# errs: optional list of float32 uncertainty arrays
periods = np.linspace(0.5, 10.0, 5000, dtype=np.float32)
period_dts = np.array([0.0], dtype=np.float32)

# Get best-period statistics
stats = bls.calc(times, mags, periods, period_dts, errs=errs, output="stats")
print(stats[0].params[0])  # detected period

# Get full periodogram
pgrams = bls.calc(times, mags, periods, period_dts, output="periodogram")

# Get top-N peaks (memory-efficient for large grids)
peaks = bls.calc(times, mags, periods, period_dts, output="peaks", n_peaks=32)
```

### Fourier Decomposition Usage

```python
import numpy as np
import periodfind

fd = periodfind.FourierDecomposition()

# times, mags, errs: lists of float32 arrays (one per light curve)
# periods: float32 array with one period per curve
features = fd.calc(times, mags, errs, periods)
# features.shape == (n_curves, 14)
```

## Throughput Benchmarks

Measured on a batch of **100 light curves** over **1,000 trial periods** (single `period_dt`). CPU = Rust/Rayon on 2x Intel Xeon E5-2680 v4 (28 cores); GPU = NVIDIA Tesla P100 (12 GB). Times are median of 3 runs after warmup.

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

### Throughput plot (log-log scale)

![Throughput benchmark](docs/throughput_points.png)

Solid lines = GPU (CUDA), dashed lines = CPU (Rust). The GPU advantage grows with light curve length for most algorithms, with LS showing the largest speedup (up to 216x). A separate curve-count scaling sweep (1–512 curves at 1,024 pts/curve) shows GPU throughput plateauing once the device is fully occupied.

See the [full benchmarks page](https://zwickytransientfacility.github.io/periodfind/benchmarks/) for curve-scaling results and methodology.

To reproduce, run `python benchmarks/throughput_bench.py` followed by `python benchmarks/plot_throughput.py`.

## Installing

### GPU backend (CUDA)

Requires CUDA installed with `nvcc` on your `PATH` (or set `$CUDA_HOME`).

```bash
pip install cython numpy
pip install -e .
```

### CPU backend (Rust)

Requires a Rust toolchain and [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
cd rust && maturin develop --release
```

This builds the `periodfind.cpu` module using Rayon for multithreaded parallelism. No GPU needed.

### Python API

Ensure that `Cython` and `numpy` are both installed. Then, simply run:

```bash
python setup.py install
```

And periodfind should be installed!

### C++ API

First, ensure that CMake is installed, and that it is at least version `3.8`. Next, create a build directory for CMake to use, and `cd` into it:

```bash
mkdir cmakebuild
cd cmakebuild
```

Now, run CMake, and build the library:

```bash
cmake ..
make
```

Finally, install the package by running `make install` (may require super-user priveleges), which will install the library in `/usr/local/lib/` and the headers in `/usr/local/include/periodfind/` by default (on Linux, location will be different on other operating systems).

## Testing

Run the full test suite with pytest:

```bash
pytest tests/ -v
```

Tests are organized into four categories:

- **Unit tests** (`test_periodfind.py`): Statistics, Periodogram, and utility tests (no GPU or Rust needed)
- **CPU standalone tests** (`test_cpu_standalone.py`): Tests for the Rust CPU backend (period-finding algorithms)
- **Fourier tests** (`test_fourier.py`): Tests for Fourier decomposition (output shape, known signal recovery, edge cases, input validation)
- **GPU integration tests** (`test_cpu_vs_cuda.py`): CUDA algorithm tests (auto-skipped if no GPU is available)

To run only CPU tests (no GPU required):

```bash
pytest tests/test_periodfind.py tests/test_cpu_standalone.py tests/test_fourier.py -v
```

## CI

GitHub Actions runs CPU tests automatically on every push and PR. See `.github/workflows/tests.yml`. GPU tests run on self-hosted runners when available.

## Compatibility

This package has been tested only on Linux hosts running CUDA 10.2 and CUDA 11. Other operating systems and versions of CUDA may work, but it is not guaranteed.

## Acknowledgements

Funding for this project was provided by the Larson Scholar Fellowship as part of the SURF program.

## License

This package is licensed under the BSD 3-clause license. The copyright holder is the California Institute of Technology (Caltech).

`setup.py` and `MANIFEST.in` are based off of an example project at <https://github.com/rmcgibbo/npcuda-example/>, licensed under the BSD 2-clause license.
