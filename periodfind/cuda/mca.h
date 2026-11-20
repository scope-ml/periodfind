// Copyright 2026. Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

#ifndef PERIODFIND_CUDA_MCA_H_
#define PERIODFIND_CUDA_MCA_H_

#include <cstddef>
#include <cstdint>
#include <vector>

// Multi-criterion ascent (MCA) period finder, a port of the CPU implementation
// in rust/src/mca.rs with the same terms and order of operations. Walkers start
// at candidate periods and climb one score built from four z-scored terms
// (two-harmonic power, robust residual scale, kernel phase dispersion and
// leave-one-night-out prediction error). Double precision throughout.
class MultiCriterionAscent {
   public:
    MultiCriterionAscent(double f_max, double oversample, double tol, int max_iter, int top);

    // One row per curve: n_starts, n_optima, then the `top` best periods and
    // their scores, NaN padded.
    void CalcMCABatched(const std::vector<double*>& times,
                        const std::vector<double*>& mags,
                        const std::vector<uint8_t*>& bands,
                        const std::vector<size_t>& lengths,
                        const std::vector<double*>& periods,
                        const std::vector<size_t>& n_periods,
                        double* out) const;

   private:
    double f_max_, oversample_, tol_;
    int max_iter_, top_;
};

#endif  // PERIODFIND_CUDA_MCA_H_
