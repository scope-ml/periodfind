#cython: language_level=3

"""
Provides an interface for fitting a bank of sampled multiband templates to
light curves (CUDA backend).

Templates are held as samples on a phase grid, as computed, rather than as a
truncated Fourier series. The distinction matters for sharp features: a series
truncated to eleven coefficients reproduces a detached eclipsing binary's
eclipse to only about 39 per cent of its own depth, and the blurred template
then fits a wrong period convincingly.

Everything is double precision. The fit is an argmin over n_template * n_phase
nearly tied columns and folds a long baseline at short periods; single
precision changes which column wins.
"""

import numpy as np

cimport numpy as np
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t
from libcpp.vector cimport vector

np.import_array()

cdef extern from "./cuda/tfs.h":
    cdef cppclass CppTemplateFitSampled "TemplateFitSampled":
        CppTemplateFitSampled(const double* samples,
                              size_t n_template,
                              size_t n_band,
                              size_t n_phase)

        void CalcTFSBatched(const vector[double*]& times,
                            const vector[double*]& mags,
                            const vector[uint8_t*]& bands,
                            const vector[size_t]& lengths,
                            const vector[double*]& periods,
                            const vector[size_t]& n_periods,
                            size_t min_points,
                            double* out) const


cdef class TemplateFitSampled:
    """Multiband sampled template bank fitting (GPU backend).

    The model at a trial period P, for band b:

        mag_b(t) = offset_b + a * S[k, b]((t/P - phi) mod 1)

    The offset is free per band, ``a`` is one amplitude shared across all
    bands, and ``phi`` runs over every point of the phase grid. Locking the band
    amplitudes together is the point: it makes colour information the fit has to
    respect rather than a free parameter per band.

    Score is the fraction of variance unexplained, pooled over bands. Every
    template costs the same parameters, so there is nothing for an information
    criterion to penalise.

    Each phase shift is an index into the grid, so all ``n_phase`` of them come
    out of one transform rather than being looped over. The light curve is
    binned onto that same grid, which is the one approximation here: a bin is
    1 / n_phase of a cycle wide.

    Parameters
    ----------
    samples : ndarray, shape (n_template, n_band, n_phase)
        Each template's magnitude against phase, in each band. ``n_phase`` must
        be a power of two. A constant offset per band is fitted, so the zero
        point of each template does not matter.
    """

    cdef CppTemplateFitSampled* tfs
    cdef readonly size_t n_template
    cdef readonly size_t n_band
    cdef readonly size_t n_phase

    def __cinit__(self, samples):
        samples = np.ascontiguousarray(samples, dtype=np.float64)
        if samples.ndim != 3:
            raise ValueError("samples must be (n_template, n_band, n_phase)")
        self.n_template = samples.shape[0]
        self.n_band = samples.shape[1]
        self.n_phase = samples.shape[2]
        if self.n_phase < 4 or (self.n_phase & (self.n_phase - 1)) != 0:
            raise ValueError("n_phase must be a power of two, at least 4")

        cdef np.ndarray[ndim=1, dtype=np.float64_t] flat = \
            np.ascontiguousarray(samples.ravel(), dtype=np.float64)

        self.tfs = new CppTemplateFitSampled(&flat[0], self.n_template,
                                             self.n_band, self.n_phase)

    def __dealloc__(self):
        if self.tfs is not NULL:
            del self.tfs

    def calc(self, list times, list mags, list bands, list periods,
             size_t min_points=12):
        """Fit every light curve at each of its own trial periods.

        Each curve brings its own periods because in practice they are a
        periodogram's top candidates rather than a shared grid.

        Parameters
        ----------
        times, mags : list of ndarray, float64
            One array per light curve, in days and magnitudes.
        bands : list of ndarray, uint8
            Band index per point, 0 to n_band - 1.
        periods : list of ndarray, float64
            Trial periods for that curve, in days.
        min_points : int, default=12
            A band with fewer points than this is skipped.

        Returns
        -------
        ndarray, shape (n_curves, 5)
            fvu, period, template, shift, amplitude. The shift is an index into
            the phase grid, so the fitted phase is shift / n_phase of a cycle.
            A curve with no usable band gives NaN and -1.
        """
        cdef size_t n = len(times)
        if not (len(mags) == n and len(bands) == n and len(periods) == n):
            raise ValueError("times, mags, bands and periods must be the same "
                             "length")

        # hold references so the buffers outlive the call
        t_arrs = [np.ascontiguousarray(a, dtype=np.float64) for a in times]
        m_arrs = [np.ascontiguousarray(a, dtype=np.float64) for a in mags]
        b_arrs = [np.ascontiguousarray(a, dtype=np.uint8) for a in bands]
        p_arrs = [np.ascontiguousarray(a, dtype=np.float64) for a in periods]

        cdef vector[double*] t_ptrs
        cdef vector[double*] m_ptrs
        cdef vector[uint8_t*] b_ptrs
        cdef vector[double*] p_ptrs
        cdef vector[size_t] lengths
        cdef vector[size_t] n_periods

        cdef np.ndarray[ndim=1, dtype=np.float64_t] ta
        cdef np.ndarray[ndim=1, dtype=np.float64_t] ma
        cdef np.ndarray[ndim=1, dtype=np.uint8_t] ba
        cdef np.ndarray[ndim=1, dtype=np.float64_t] pa

        cdef size_t i
        for i in range(n):
            ta = t_arrs[i]
            ma = m_arrs[i]
            ba = b_arrs[i]
            pa = p_arrs[i]
            if ta.shape[0] != ma.shape[0] or ta.shape[0] != ba.shape[0]:
                raise ValueError("curve %d: times, mags and bands differ in "
                                 "length" % i)
            if ta.shape[0] == 0 or pa.shape[0] == 0:
                raise ValueError("curve %d is empty" % i)
            t_ptrs.push_back(&ta[0])
            m_ptrs.push_back(&ma[0])
            b_ptrs.push_back(&ba[0])
            p_ptrs.push_back(&pa[0])
            lengths.push_back(ta.shape[0])
            n_periods.push_back(pa.shape[0])

        cdef np.ndarray[ndim=2, dtype=np.float64_t] out = \
            np.zeros((n, 5), dtype=np.float64)

        self.tfs.CalcTFSBatched(t_ptrs, m_ptrs, b_ptrs, lengths, p_ptrs,
                                n_periods, min_points, &out[0, 0])
        return out
