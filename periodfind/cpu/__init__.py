"""CPU implementations of period-finding algorithms.

Provides the same API as the CUDA-backed Cython wrappers (ce.pyx, aov.pyx,
ls.pyx), but uses a Rust+Rayon backend that runs on the CPU.
"""

import numpy as np
from periodfind_cpu import (
    calc_aov_batched,
    calc_aov_peaks_batched,
    calc_ce_batched,
    calc_ce_peaks_batched,
    calc_fourier_batched,
    calc_fpw_batched,
    calc_fpw_peaks_batched,
    calc_ls_batched,
    calc_ls_peaks_batched,
    find_top_peaks_batched,  # noqa: F401  (public API)
)

from periodfind import Periodogram, Statistics
from periodfind._utils import ensure_float32, prepare_magnitudes, validate_inputs

N_PEAKS_DEFAULT = 32
MIN_DISTANCE_DEFAULT = 1


def _unravel_peaks(peak_indices, peak_values, periods, period_dts, n_peaks):
    """Convert flat peak indices + values into a list of lists of Statistics."""
    n_curves = peak_indices.shape[0]
    n_pdts = len(period_dts)
    results = []
    for ci in range(n_curves):
        curve_peaks = []
        for pi in range(n_peaks):
            flat_idx = peak_indices[ci, pi]
            if flat_idx < 0:
                break
            p_idx = int(flat_idx // n_pdts)
            d_idx = int(flat_idx % n_pdts)
            curve_peaks.append(
                Statistics(
                    params=[float(periods[p_idx]), float(period_dts[d_idx])],
                    value=float(peak_values[ci, pi]),
                    mean=0.0,
                    std=1.0,
                    median=0.0,
                    mad=1.0,
                )
            )
        results.append(curve_peaks)
    return results


class ConditionalEntropy:
    """Conditional Entropy based light curve analysis (CPU backend).

    Parameters
    ----------
    n_phase : int, default=10
        The number of phase bins in the histogram.
    n_mag : int, default=10
        The number of magnitude bins in the histogram.
    phase_bin_extent : int, default=1
        Effective width (in bins) of each phase bin (overlap/smoothing).
    mag_bin_extent : int, default=1
        Effective width (in bins) of each magnitude bin (overlap/smoothing).
    """

    def __init__(self, n_phase=10, n_mag=10, phase_bin_extent=1, mag_bin_extent=1):
        self.n_phase = n_phase
        self.n_mag = n_mag
        self.phase_bin_extent = phase_bin_extent
        self.mag_bin_extent = mag_bin_extent

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        output="stats",
        normalize=True,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Conditional Entropy calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.  ``'peaks'`` uses a streaming greedy
            algorithm to return the top *n_peaks* without materialising the
            full 3-D periodogram array, saving memory on large grids.
        normalize : bool, default=True
            Whether to normalize magnitudes to (0, 1).
        center : bool, default=False
            Whether to center magnitudes to zero mean.
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance (in flattened period grid samples) between
            accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        mags_use = prepare_magnitudes(mags, center, normalize)

        if output == "peaks":
            idx, val = calc_ce_peaks_batched(
                times,
                mags_use,
                periods,
                period_dts,
                self.n_phase,
                self.n_mag,
                self.phase_bin_extent,
                self.mag_bin_extent,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        ces_ndarr = calc_ce_batched(
            times,
            mags_use,
            periods,
            period_dts,
            self.n_phase,
            self.n_mag,
            self.phase_bin_extent,
            self.mag_bin_extent,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    ces_ndarr[i],
                    [periods, period_dts],
                    False,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], False) for data in ces_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class AOV:
    """Analysis-of-Variance based light curve analysis (CPU backend).

    Parameters
    ----------
    n_phase : int, default=10
        The number of phase bins.
    phase_bin_extent : int, default=1
        Effective width (in bins) of each phase bin (overlap/smoothing).
    """

    def __init__(self, n_phase=10, phase_bin_extent=1):
        self.n_phase = n_phase
        self.phase_bin_extent = phase_bin_extent

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        output="stats",
        normalize=False,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Analysis-of-Variance calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Whether to normalize magnitudes to (0, 1).
        center : bool, default=False
            Whether to center magnitudes to zero mean.
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        mags_use = prepare_magnitudes(mags, center, normalize)

        if output == "peaks":
            idx, val = calc_aov_peaks_batched(
                times,
                mags_use,
                periods,
                period_dts,
                self.n_phase,
                self.phase_bin_extent,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        aovs_ndarr = calc_aov_batched(
            times,
            mags_use,
            periods,
            period_dts,
            self.n_phase,
            self.phase_bin_extent,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    aovs_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in aovs_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class LombScargle:
    """Lomb-Scargle periodogram light curve analysis (CPU backend)."""

    def __init__(self):
        pass

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        output="stats",
        normalize=False,
        center=True,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Lomb-Scargle calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Whether to normalize magnitudes to (0, 1).
        center : bool, default=True
            Whether to center magnitudes to zero mean.
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        mags_use = prepare_magnitudes(mags, center, normalize)

        if output == "peaks":
            idx, val = calc_ls_peaks_batched(
                times,
                mags_use,
                periods,
                period_dts,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        ls_ndarr = calc_ls_batched(
            times,
            mags_use,
            periods,
            period_dts,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    ls_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in ls_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class FPW:
    """Fast Phase-folding Weighted (FPW) light curve analysis (CPU backend).

    Computes the FPW statistic (Finkbeiner et al. 2025), a weighted
    chi-squared reduction that supports per-point uncertainties.

    Parameters
    ----------
    n_bins : int, default=10
        The number of phase bins.
    """

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        errs=None,
        output="stats",
        normalize=False,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs FPW calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        errs : list of ndarray or None, default=None
            List of per-point uncertainties (standard deviations).
            If None, uniform uncertainties of 1.0 are assumed.
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Unused (accepted for API consistency).
        center : bool, default=False
            Unused (accepted for API consistency).
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        # Handle uncertainties
        if errs is None:
            errs = [np.ones(len(t), dtype=np.float32) for t in times]
        else:
            ensure_float32(errs, "errs")
            if len(errs) != len(times):
                raise ValueError(
                    f"errs must have the same number of arrays as times, "
                    f"got {len(errs)} and {len(times)}"
                )
            for i, (e, t) in enumerate(zip(errs, times)):
                if len(e) != len(t):
                    raise ValueError(
                        f"errs[{i}] and times[{i}] have different lengths: {len(e)} vs {len(t)}"
                    )

        if output == "peaks":
            idx, val = calc_fpw_peaks_batched(
                times,
                mags,
                errs,
                periods,
                period_dts,
                self.n_bins,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        fpw_ndarr = calc_fpw_batched(
            times,
            mags,
            errs,
            periods,
            period_dts,
            self.n_bins,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    fpw_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in fpw_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class FourierDecomposition:
    """Fourier decomposition via weighted linear least-squares (CPU backend).

    Computes Fourier features for light curves given pre-determined periods.
    Uses BIC model selection over 0–5 harmonics.

    Returns 14 features per curve:
        [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
    """

    def calc(self, times, mags, errs, periods):
        """Run Fourier decomposition on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times (float32).
        mags : list of ndarray
            List of light curve magnitudes (float32).
        errs : list of ndarray
            List of per-point uncertainties (float32).
        periods : ndarray
            Array of periods, one per curve (float32).

        Returns
        -------
        ndarray of shape (n_curves, 14)
            Fourier features for each curve.
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")
        ensure_float32(errs, "errs")

        if len(errs) != len(times):
            raise ValueError(
                f"errs must have the same number of arrays as times, "
                f"got {len(errs)} and {len(times)}"
            )
        for i, (e, t) in enumerate(zip(errs, times)):
            if len(e) != len(t):
                raise ValueError(
                    f"errs[{i}] and times[{i}] have different lengths: {len(e)} vs {len(t)}"
                )

        periods = np.asarray(periods, dtype=np.float32)
        if periods.ndim != 1 or len(periods) != len(times):
            raise ValueError(
                f"periods must be a 1D array with one entry per curve, "
                f"got shape {periods.shape} for {len(times)} curves"
            )

        return calc_fourier_batched(times, mags, errs, periods)
