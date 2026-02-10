"""Cross-backend comparison tests: CPU (Rust) vs CUDA.

Verifies that the CPU and CUDA backends produce numerically equivalent
periodograms for all three algorithms. Requires a GPU.

Run with: pytest tests/test_cpu_vs_cuda.py -v
"""

import numpy as np
import pytest
import subprocess

# Check GPU availability
HAS_GPU = False
try:
    from periodfind.ce import ConditionalEntropy as CudaCE
    from periodfind.aov import AOV as CudaAOV
    from periodfind.ls import LombScargle as CudaLS
    ret = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
    if ret.returncode == 0:
        HAS_GPU = True
except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
    pass

# CPU backend should always be importable
from periodfind.cpu import (
    ConditionalEntropy as CpuCE,
    AOV as CpuAOV,
    LombScargle as CpuLS,
)

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CUDA GPU not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sinusoidal_lightcurve(period, n_points=500, amplitude=1.0,
                                noise_std=0.05, t_span=100.0, seed=42):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = 2.0 * np.pi * times / period
    mags = (amplitude * np.sin(phase) + amplitude).astype(np.float32)
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def make_eclipsing_binary(period, n_points=500, eclipse_depth=0.5,
                          eclipse_width=0.1, noise_std=0.02,
                          t_span=200.0, seed=42):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = (times / period) % 1.0
    dist = np.minimum(phase, 1.0 - phase)
    mags = np.ones(n_points, dtype=np.float32)
    in_eclipse = dist < eclipse_width / 2
    mags[in_eclipse] = 1.0 - eclipse_depth * (1.0 - dist[in_eclipse]
                                                / (eclipse_width / 2))
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def make_rr_lyrae(period, n_points=500, amplitude=0.8, noise_std=0.02,
                  t_span=200.0, seed=42):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = (times / period) % 1.0
    rise_end = 0.2
    mags = np.where(
        phase < rise_end,
        amplitude * (phase / rise_end),
        amplitude * (1.0 - (phase - rise_end) / (1.0 - rise_end))
    ).astype(np.float32)
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def make_trial_periods(true_period, n_periods=200, margin=0.5):
    lo = max(true_period - margin * true_period, 0.01)
    hi = true_period + margin * true_period
    return np.linspace(lo, hi, n_periods, dtype=np.float32)


# ---------------------------------------------------------------------------
# Conditional Entropy: CPU vs CUDA
# ---------------------------------------------------------------------------

@requires_gpu
class TestCECpuVsCuda:
    def test_sinusoidal_agreement(self):
        """CE periodograms should match for a sinusoidal signal."""
        t, m = make_sinusoidal_lightcurve(period=5.0, n_points=800,
                                           noise_std=0.02, t_span=200.0)
        periods = make_trial_periods(5.0, n_periods=300)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_ce = CudaCE(n_phase=10, n_mag=10)
        cpu_ce = CpuCE(n_phase=10, n_mag=10)

        cuda_pgram = cuda_ce.calc([t], [m], periods, period_dts,
                                   output='periodogram')[0].data
        cpu_pgram = cpu_ce.calc([t], [m], periods, period_dts,
                                 output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_eclipsing_binary_agreement(self):
        """CE periodograms should match for an eclipsing binary."""
        t, m = make_eclipsing_binary(period=2.5, n_points=600)
        periods = make_trial_periods(2.5, n_periods=200)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_ce = CudaCE(n_phase=15, n_mag=10)
        cpu_ce = CpuCE(n_phase=15, n_mag=10)

        cuda_pgram = cuda_ce.calc([t], [m], periods, period_dts,
                                   output='periodogram')[0].data
        cpu_pgram = cpu_ce.calc([t], [m], periods, period_dts,
                                 output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_multiple_period_dts(self):
        """CE should match with multiple period derivatives."""
        t, m = make_sinusoidal_lightcurve(period=3.0, n_points=500)
        periods = make_trial_periods(3.0, n_periods=100)
        period_dts = np.linspace(-0.01, 0.01, 5, dtype=np.float32)

        cuda_ce = CudaCE(n_phase=10, n_mag=10)
        cpu_ce = CpuCE(n_phase=10, n_mag=10)

        cuda_pgram = cuda_ce.calc([t], [m], periods, period_dts,
                                   output='periodogram')[0].data
        cpu_pgram = cpu_ce.calc([t], [m], periods, period_dts,
                                 output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_batched_agreement(self):
        """CE should match for batched light curves."""
        lcs = [make_sinusoidal_lightcurve(period=p, seed=i)
               for i, p in enumerate([2.0, 4.0, 6.0])]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_ce = CudaCE(n_phase=10, n_mag=10)
        cpu_ce = CpuCE(n_phase=10, n_mag=10)

        cuda_results = cuda_ce.calc(times, mag_list, periods, period_dts,
                                     output='periodogram')
        cpu_results = cpu_ce.calc(times, mag_list, periods, period_dts,
                                   output='periodogram')

        for i in range(3):
            np.testing.assert_allclose(cpu_results[i].data, cuda_results[i].data,
                                        rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# AOV: CPU vs CUDA
# ---------------------------------------------------------------------------

@requires_gpu
class TestAOVCpuVsCuda:
    def test_sinusoidal_agreement(self):
        """AOV periodograms should match for a sinusoidal signal."""
        t, m = make_sinusoidal_lightcurve(period=5.0, n_points=800,
                                           noise_std=0.02, t_span=200.0)
        periods = make_trial_periods(5.0, n_periods=300)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_aov = CudaAOV(n_phase=10)
        cpu_aov = CpuAOV(n_phase=10)

        cuda_pgram = cuda_aov.calc([t], [m], periods, period_dts,
                                    output='periodogram')[0].data
        cpu_pgram = cpu_aov.calc([t], [m], periods, period_dts,
                                  output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_rr_lyrae_agreement(self):
        """AOV periodograms should match for an RR Lyrae signal."""
        t, m = make_rr_lyrae(period=0.6, n_points=600, t_span=50.0)
        periods = make_trial_periods(0.6, n_periods=200)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_aov = CudaAOV(n_phase=15)
        cpu_aov = CpuAOV(n_phase=15)

        cuda_pgram = cuda_aov.calc([t], [m], periods, period_dts,
                                    output='periodogram')[0].data
        cpu_pgram = cpu_aov.calc([t], [m], periods, period_dts,
                                  output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_multiple_period_dts(self):
        """AOV should match with multiple period derivatives."""
        t, m = make_sinusoidal_lightcurve(period=3.0, n_points=500)
        periods = make_trial_periods(3.0, n_periods=100)
        period_dts = np.linspace(-0.01, 0.01, 5, dtype=np.float32)

        cuda_aov = CudaAOV(n_phase=10)
        cpu_aov = CpuAOV(n_phase=10)

        cuda_pgram = cuda_aov.calc([t], [m], periods, period_dts,
                                    output='periodogram')[0].data
        cpu_pgram = cpu_aov.calc([t], [m], periods, period_dts,
                                  output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_batched_agreement(self):
        """AOV should match for batched light curves."""
        lcs = [make_sinusoidal_lightcurve(period=p, seed=i)
               for i, p in enumerate([2.0, 4.0, 6.0])]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_aov = CudaAOV(n_phase=10)
        cpu_aov = CpuAOV(n_phase=10)

        cuda_results = cuda_aov.calc(times, mag_list, periods, period_dts,
                                      output='periodogram')
        cpu_results = cpu_aov.calc(times, mag_list, periods, period_dts,
                                    output='periodogram')

        for i in range(3):
            np.testing.assert_allclose(cpu_results[i].data, cuda_results[i].data,
                                        rtol=1e-4, atol=1e-6)

    def test_overlap_agreement(self):
        """AOV with bin overlap should match between backends."""
        t, m = make_sinusoidal_lightcurve(period=4.0, n_points=600)
        periods = make_trial_periods(4.0, n_periods=200)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_aov = CudaAOV(n_phase=10, phase_bin_extent=3)
        cpu_aov = CpuAOV(n_phase=10, phase_bin_extent=3)

        cuda_pgram = cuda_aov.calc([t], [m], periods, period_dts,
                                    output='periodogram')[0].data
        cpu_pgram = cpu_aov.calc([t], [m], periods, period_dts,
                                  output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Lomb-Scargle: CPU vs CUDA
# ---------------------------------------------------------------------------

@requires_gpu
class TestLSCpuVsCuda:
    def test_sinusoidal_agreement(self):
        """LS periodograms should match for a sinusoidal signal."""
        t, m = make_sinusoidal_lightcurve(period=5.0, n_points=800,
                                           noise_std=0.02, t_span=200.0)
        periods = make_trial_periods(5.0, n_periods=300)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_ls = CudaLS()
        cpu_ls = CpuLS()

        cuda_pgram = cuda_ls.calc([t], [m], periods, period_dts,
                                   output='periodogram')[0].data
        cpu_pgram = cpu_ls.calc([t], [m], periods, period_dts,
                                 output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_eclipsing_binary_agreement(self):
        """LS periodograms should match for an eclipsing binary."""
        t, m = make_eclipsing_binary(period=2.5, n_points=600)
        periods = make_trial_periods(2.5, n_periods=200)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_ls = CudaLS()
        cpu_ls = CpuLS()

        cuda_pgram = cuda_ls.calc([t], [m], periods, period_dts,
                                   output='periodogram')[0].data
        cpu_pgram = cpu_ls.calc([t], [m], periods, period_dts,
                                 output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_rr_lyrae_agreement(self):
        """LS periodograms should match for an RR Lyrae signal."""
        t, m = make_rr_lyrae(period=0.6, n_points=600, t_span=50.0)
        periods = make_trial_periods(0.6, n_periods=200)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_ls = CudaLS()
        cpu_ls = CpuLS()

        cuda_pgram = cuda_ls.calc([t], [m], periods, period_dts,
                                   output='periodogram')[0].data
        cpu_pgram = cpu_ls.calc([t], [m], periods, period_dts,
                                 output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_multiple_period_dts(self):
        """LS should match with multiple period derivatives."""
        t, m = make_sinusoidal_lightcurve(period=3.0, n_points=500)
        periods = make_trial_periods(3.0, n_periods=100)
        period_dts = np.linspace(-0.01, 0.01, 5, dtype=np.float32)

        cuda_ls = CudaLS()
        cpu_ls = CpuLS()

        cuda_pgram = cuda_ls.calc([t], [m], periods, period_dts,
                                   output='periodogram')[0].data
        cpu_pgram = cpu_ls.calc([t], [m], periods, period_dts,
                                 output='periodogram')[0].data

        np.testing.assert_allclose(cpu_pgram, cuda_pgram, rtol=1e-4, atol=1e-6)

    def test_batched_agreement(self):
        """LS should match for batched light curves."""
        lcs = [make_sinusoidal_lightcurve(period=p, seed=i)
               for i, p in enumerate([2.0, 4.0, 6.0])]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        cuda_ls = CudaLS()
        cpu_ls = CpuLS()

        cuda_results = cuda_ls.calc(times, mag_list, periods, period_dts,
                                     output='periodogram')
        cpu_results = cpu_ls.calc(times, mag_list, periods, period_dts,
                                   output='periodogram')

        for i in range(3):
            np.testing.assert_allclose(cpu_results[i].data, cuda_results[i].data,
                                        rtol=1e-4, atol=1e-6)
