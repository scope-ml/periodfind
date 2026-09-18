// Copyright 2026. Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

#ifndef PERIODFIND_CUDA_TFS_H_
#define PERIODFIND_CUDA_TFS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

// Template fitting on sampled templates.
//
// Same fit as TemplateFit, with the bank held as samples on a phase grid
// instead of a truncated Fourier series. A truncated series cannot hold a
// narrow eclipse: at eleven coefficients a detached eclipsing binary is
// reproduced to about 39 per cent of its own eclipse depth, and the blurred
// template then fits a wrong period convincingly.
//
// Every sum the fit needs is a circular correlation between the folded light
// curve and the template, so one transform yields the score at every phase
// shift at once. The transforms of S and S^2 belong to the bank and are taken
// once, when the bank is uploaded.
//
// Double precision throughout: the score is an argmin over n_template *
// n_phase nearly tied columns, and single precision changes which one wins.
class TemplateFitSampled {
   public:
    // samples: bank laid out [template][band][phase], n_phase a power of two.
    TemplateFitSampled(const double* samples,
                       size_t n_template,
                       size_t n_band,
                       size_t n_phase);
    ~TemplateFitSampled();

    size_t NumTemplate() const { return n_template_; }
    size_t NumBand() const { return n_band_; }
    size_t NumPhase() const { return n_phase_; }

    // One row per curve: fvu, period, template, shift, amplitude. A curve with
    // no usable band gives NaN and -1. The shift is an index into the phase
    // grid, so the fitted phase is shift / n_phase of a cycle.
    void CalcTFSBatched(const std::vector<double*>& times,
                        const std::vector<double*>& mags,
                        const std::vector<uint8_t*>& bands,
                        const std::vector<size_t>& lengths,
                        const std::vector<double*>& periods,
                        const std::vector<size_t>& n_periods,
                        size_t min_points,
                        double* out) const;

   private:
    size_t n_template_;
    size_t n_band_;
    size_t n_phase_;
    // transforms of S and of S^2, [template][band][phase], interleaved re/im
    double* dev_fs_;
    double* dev_fs2_;
};

#endif  // PERIODFIND_CUDA_TFS_H_
