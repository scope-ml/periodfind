/// Shared time-folding logic used by all three algorithms.
///
/// Matches the CUDA implementation exactly:
///   t_corr = t - pdt_corr * t * t
///   folded = |fract(t_corr / period)|

/// Compute the period derivative correction factor.
#[inline(always)]
pub fn pdt_correction(period: f32, period_dt: f32) -> f32 {
    (period_dt / period) / 2.0
}

/// Fold a time value into the [0, 1) phase range with period derivative correction.
#[inline(always)]
pub fn fold_time(t: f32, period: f32, pdt_corr: f32) -> f32 {
    let t_corr = t - pdt_corr * t * t;
    let ratio = t_corr / period;
    (ratio - ratio.floor()).abs()
}
