#cython: language_level=3

"""
Provides an interface to the multi-criterion ascent (MCA) period finder (CUDA
backend). Same method and results as ``periodfind.cpu.MultiCriterionAscent``,
up to the last digits of the GPU's trigonometric functions.
"""

import numpy as np

cimport numpy as np
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t
from libcpp.vector cimport vector

np.import_array()

cdef extern from "./cuda/mca.h":
    cdef cppclass CppMultiCriterionAscent "MultiCriterionAscent":
        CppMultiCriterionAscent(double f_max, double oversample, double tol,
                                int max_iter, int top)

        void CalcMCABatched(const vector[double*]& times,
                            const vector[double*]& mags,
                            const vector[uint8_t*]& bands,
                            const vector[size_t]& lengths,
                            const vector[double*]& periods,
                            const vector[size_t]& n_periods,
                            double* out) const


cdef class MultiCriterionAscent:
    """Multi-criterion ascent (MCA) period finder (GPU backend).

    See ``periodfind.cpu.MultiCriterionAscent`` for the method and parameters;
    the arguments and output are the same.
    """

    cdef CppMultiCriterionAscent* mca
    cdef readonly double f_max
    cdef readonly double oversample
    cdef readonly double tol
    cdef readonly int max_iter
    cdef readonly int top

    def __cinit__(self, f_max=288.0, oversample=5, tol=1e-3, max_iter=10000, top=50):
        if not (np.isfinite(f_max) and f_max > 0):
            raise ValueError("f_max must be positive, got %r" % (f_max,))
        if oversample < 1 or tol <= 0 or max_iter < 1 or top < 1:
            raise ValueError("oversample >= 1, tol > 0, max_iter >= 1 and top >= 1 are required")
        self.f_max = f_max
        self.oversample = oversample
        self.tol = tol
        self.max_iter = max_iter
        self.top = top
        self.mca = new CppMultiCriterionAscent(self.f_max, self.oversample, self.tol,
                                               self.max_iter, self.top)

    def __dealloc__(self):
        if self.mca is not NULL:
            del self.mca

    @property
    def columns(self):
        """Names of the output columns."""
        return (["n_starts", "n_optima"]
                + ["period_%d" % k for k in range(1, self.top + 1)]
                + ["score_%d" % k for k in range(1, self.top + 1)])

    def calc(self, list times, list mags, list bands, list periods):
        """Search every light curve from its own candidate periods.

        Returns an ndarray of shape (n_curves, 2 + 2 * top): n_starts,
        n_optima, then the ``top`` best periods (best first) and their scores.
        Missing entries are NaN.
        """
        cdef size_t n = len(times)
        if not (len(mags) == n and len(bands) == n and len(periods) == n):
            raise ValueError("times, mags, bands and periods must be the same length")

        t_arrs = [np.ascontiguousarray(a, dtype=np.float64) for a in times]
        m_arrs = [np.ascontiguousarray(a, dtype=np.float64) for a in mags]
        b_arrs = [np.ascontiguousarray(a, dtype=np.uint8) for a in bands]
        p_arrs = [np.ascontiguousarray(np.atleast_1d(a), dtype=np.float64) for a in periods]

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
                raise ValueError("curve %d: times, mags and bands differ in length" % i)
            if ta.shape[0] == 0 or pa.shape[0] == 0:
                raise ValueError("curve %d is empty" % i)
            t_ptrs.push_back(&ta[0])
            m_ptrs.push_back(&ma[0])
            b_ptrs.push_back(&ba[0])
            p_ptrs.push_back(&pa[0])
            lengths.push_back(ta.shape[0])
            n_periods.push_back(pa.shape[0])

        cdef np.ndarray[ndim=2, dtype=np.float64_t] out = \
            np.zeros((n, 2 + 2 * self.top), dtype=np.float64)
        self.mca.CalcMCABatched(t_ptrs, m_ptrs, b_ptrs, lengths, p_ptrs, n_periods, &out[0, 0])
        return out
