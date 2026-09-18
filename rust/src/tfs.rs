/// Template Fit on Sampled templates (TFS), CPU.
///
/// Same fit as `tf`, different representation. `tf` stores each template as a
/// truncated Fourier series, which is exact for a smooth curve and wrong for a
/// sharp one: at eleven coefficients a detached eclipsing binary's eclipse is
/// reproduced to about 39 per cent of its own depth, and the blurred template
/// then fits a wrong period convincingly. This stores the template as it was
/// computed, as samples on a phase grid, and loses nothing.
///
/// Model at trial period P, for band b:
///
///     mag_b(t) = offset_b + a * S[k, b]((t/P - phi) mod 1)
///
/// with a free offset per band, one amplitude `a` shared across all bands, and
/// `phi` running over every point of the phase grid. Score is the fraction of
/// variance unexplained, pooled over bands, exactly as in `tf`.
///
/// # Why this is not slower
///
/// Fold the light curve onto the same grid the templates use, accumulating per
/// bin counts `c` and summed magnitudes `s`. Every sum the fit needs is then a
/// circular correlation against the template, and one transform produces the
/// value at *every* phase shift at once:
///
///     num(phi) = sum_r yc_r S(ph_r - phi) = (s * S)(phi) - mbar (c * S)(phi)
///     den(phi) = sum_r S^2 - (sum_r S)^2 / n
///              = (c * S2)(phi) - (c * S)(phi)^2 / n
///
/// where `*` is circular correlation. `S` and `S^2` are properties of the
/// template, so their transforms are computed once when the bank is built and
/// reused for every curve and every trial period. The per template cost is
/// then three inverse transforms, about the same as `tf` spends on 32 shifts,
/// except that it buys all `n_phase` of them.
use rayon::prelude::*;

/// Largest phase grid the fixed buffers allow. Must be a power of two.
const MAX_PHASE: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const ZERO: Cx = Cx { re: 0.0, im: 0.0 };

    #[inline]
    fn mul(self, o: Cx) -> Cx {
        Cx {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    /// `self * conj(o)`, which is what a cross correlation needs.
    #[inline]
    fn mul_conj(self, o: Cx) -> Cx {
        Cx {
            re: self.re * o.re + self.im * o.im,
            im: self.im * o.re - self.re * o.im,
        }
    }
}

/// In place iterative radix-2 FFT. `a.len()` must be a power of two.
///
/// Written out rather than taken from a crate because the CUDA kernel needs the
/// same transform in shared memory, and two implementations that have to agree
/// bit for bit are easier to keep honest when they are the same few lines.
fn fft(a: &mut [Cx], inverse: bool) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());

    // bit reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        let wl = Cx {
            re: ang.cos(),
            im: ang.sin(),
        };
        let mut i = 0usize;
        while i < n {
            let mut w = Cx { re: 1.0, im: 0.0 };
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = a[i + k + len / 2].mul(w);
                a[i + k] = Cx {
                    re: u.re + v.re,
                    im: u.im + v.im,
                };
                a[i + k + len / 2] = Cx {
                    re: u.re - v.re,
                    im: u.im - v.im,
                };
                w = w.mul(wl);
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv = 1.0 / n as f64;
        for x in a.iter_mut() {
            x.re *= inv;
            x.im *= inv;
        }
    }
}

/// A bank of sampled templates, held as the transforms the fit actually uses.
///
/// Laid out [template][band][phase]. The samples themselves are not kept: only
/// the transforms of `S` and of `S^2` are ever needed.
pub struct Bank {
    pub n_template: usize,
    pub n_band: usize,
    pub n_phase: usize,
    fs: Vec<Cx>,
    fs2: Vec<Cx>,
}

impl Bank {
    /// Build from samples laid out [template][band][phase].
    pub fn new(samples: &[f64], n_template: usize, n_band: usize, n_phase: usize) -> Self {
        assert_eq!(
            samples.len(),
            n_template * n_band * n_phase,
            "samples must be n_template * n_band * n_phase"
        );
        assert!(
            n_phase.is_power_of_two(),
            "n_phase {n_phase} must be a power of two"
        );
        assert!(
            n_phase <= MAX_PHASE,
            "n_phase {n_phase} exceeds the compiled limit {MAX_PHASE}"
        );
        assert!(n_phase >= 4, "n_phase {n_phase} is too small to fold onto");

        let mut fs = vec![Cx::ZERO; samples.len()];
        let mut fs2 = vec![Cx::ZERO; samples.len()];
        let mut buf = vec![Cx::ZERO; n_phase];
        for kb in 0..n_template * n_band {
            let off = kb * n_phase;
            for p in 0..n_phase {
                buf[p] = Cx {
                    re: samples[off + p],
                    im: 0.0,
                };
            }
            fft(&mut buf, false);
            fs[off..off + n_phase].copy_from_slice(&buf);

            for p in 0..n_phase {
                let s = samples[off + p];
                buf[p] = Cx { re: s * s, im: 0.0 };
            }
            fft(&mut buf, false);
            fs2[off..off + n_phase].copy_from_slice(&buf);
        }
        Bank {
            n_template,
            n_band,
            n_phase,
            fs,
            fs2,
        }
    }
}

/// The winning fit for one light curve.
#[derive(Debug, Clone, Copy)]
pub struct Fit {
    pub fvu: f64,
    pub period: f64,
    pub template: usize,
    pub shift: usize,
    pub amp: f64,
}

/// One band's light curve folded onto the phase grid.
struct Folded {
    /// counts per bin, transformed
    fc: Vec<Cx>,
    /// summed magnitudes per bin, transformed
    fsum: Vec<Cx>,
    n: f64,
    mean: f64,
    tss: f64,
}

/// Fold one band at a trial period, or None if it cannot contribute.
fn fold_band(
    times: &[f64],
    mags: &[f64],
    period: f64,
    n_phase: usize,
    min_points: usize,
) -> Option<Folded> {
    if times.len() < min_points {
        return None;
    }
    let mut c = vec![Cx::ZERO; n_phase];
    let mut s = vec![Cx::ZERO; n_phase];
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    for r in 0..times.len() {
        let ph = (times[r] / period).rem_euclid(1.0);
        let mut k = (ph * n_phase as f64) as usize;
        if k >= n_phase {
            k = n_phase - 1;
        }
        c[k].re += 1.0;
        s[k].re += mags[r];
        sum += mags[r];
        sumsq += mags[r] * mags[r];
    }
    let n = times.len() as f64;
    let mean = sum / n;
    let tss = sumsq - n * mean * mean;
    if tss <= 0.0 {
        return None;
    }
    fft(&mut c, false);
    fft(&mut s, false);
    Some(Folded {
        fc: c,
        fsum: s,
        n,
        mean,
        tss,
    })
}

/// Accumulate num and den over every template and shift, for one period.
#[allow(clippy::too_many_arguments)]
fn fit_period(
    per_band: &[Option<Folded>],
    bank: &Bank,
    num: &mut [f64],
    den: &mut [f64],
    work: &mut [Cx],
    tss_total: &mut f64,
) -> bool {
    let p = bank.n_phase;
    for x in num.iter_mut() {
        *x = 0.0;
    }
    for x in den.iter_mut() {
        *x = 0.0;
    }
    *tss_total = 0.0;

    let (w1, rest) = work.split_at_mut(p);
    let (w2, w3) = rest.split_at_mut(p);

    for (b, slot) in per_band.iter().enumerate() {
        let f = match slot {
            Some(f) => f,
            None => continue,
        };
        *tss_total += f.tss;
        for k in 0..bank.n_template {
            let off = (k * bank.n_band + b) * p;
            for u in 0..p {
                w1[u] = f.fsum[u].mul_conj(bank.fs[off + u]);
                w2[u] = f.fc[u].mul_conj(bank.fs[off + u]);
                w3[u] = f.fc[u].mul_conj(bank.fs2[off + u]);
            }
            fft(w1, true);
            fft(w2, true);
            fft(w3, true);
            let base = k * p;
            for tau in 0..p {
                let cs = w2[tau].re;
                num[base + tau] += w1[tau].re - f.mean * cs;
                den[base + tau] += w3[tau].re - cs * cs / f.n;
            }
        }
    }
    *tss_total > 0.0
}

/// Fit one light curve at every trial period and return the best.
pub fn fit_one(
    times: &[f64],
    mags: &[f64],
    bands: &[u8],
    periods: &[f64],
    bank: &Bank,
    min_points: usize,
) -> Fit {
    let p = bank.n_phase;
    let ncol = bank.n_template * p;
    let mut num = vec![0.0f64; ncol];
    let mut den = vec![0.0f64; ncol];
    let mut work = vec![Cx::ZERO; 3 * p];

    // split by band once; the fold itself depends on the period
    let mut bt: Vec<Vec<f64>> = (0..bank.n_band).map(|_| Vec::new()).collect();
    let mut bm: Vec<Vec<f64>> = (0..bank.n_band).map(|_| Vec::new()).collect();
    for i in 0..times.len() {
        let b = bands[i] as usize;
        if b < bank.n_band {
            bt[b].push(times[i]);
            bm[b].push(mags[i]);
        }
    }

    let mut best = Fit {
        fvu: f64::NAN,
        period: f64::NAN,
        template: usize::MAX,
        shift: usize::MAX,
        amp: f64::NAN,
    };
    let mut best_fvu = f64::INFINITY;

    for &period in periods {
        let per_band: Vec<Option<Folded>> = (0..bank.n_band)
            .map(|b| fold_band(&bt[b], &bm[b], period, p, min_points))
            .collect();
        let mut tss = 0.0;
        if !fit_period(&per_band, bank, &mut num, &mut den, &mut work, &mut tss) {
            continue;
        }
        for col in 0..ncol {
            let (nu, de) = (num[col], den[col]);
            let a = if de > 1e-14 { (nu / de).max(0.0) } else { 0.0 };
            let rss = tss - 2.0 * a * nu + a * a * de;
            let fvu = rss / tss;
            if fvu < best_fvu {
                best_fvu = fvu;
                best = Fit {
                    fvu,
                    period,
                    template: col / p,
                    shift: col % p,
                    amp: a,
                };
            }
        }
    }
    best
}

/// Fit a batch of light curves in parallel, one object per thread task.
pub fn fit_batched(
    times: &[Vec<f64>],
    mags: &[Vec<f64>],
    bands: &[Vec<u8>],
    periods: &[Vec<f64>],
    bank: &Bank,
    min_points: usize,
) -> Vec<Fit> {
    (0..times.len())
        .into_par_iter()
        .map(|i| {
            fit_one(
                &times[i],
                &mags[i],
                &bands[i],
                &periods[i],
                bank,
                min_points,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
    const P: usize = 128;

    /// A bank whose single template is a cosine, scaled per band.
    fn cosine_bank(ratios: &[f64]) -> Bank {
        let n_band = ratios.len();
        let mut s = vec![0.0; n_band * P];
        for (b, r) in ratios.iter().enumerate() {
            for k in 0..P {
                s[b * P + k] = r * (TWO_PI * k as f64 / P as f64).cos();
            }
        }
        Bank::new(&s, 1, n_band, P)
    }

    /// A cosine and a narrow dip, the shape a truncated series cannot hold.
    fn cosine_and_eclipse_bank(n_band: usize) -> Bank {
        let mut s = vec![0.0; 2 * n_band * P];
        for b in 0..n_band {
            for k in 0..P {
                let ph = k as f64 / P as f64;
                s[b * P + k] = (TWO_PI * ph).cos();
                // flat, with one bin in twelve pulled down
                s[(n_band + b) * P + k] = if ph < 1.0 / 12.0 { -1.0 } else { 0.0 };
            }
        }
        Bank::new(&s, 2, n_band, P)
    }

    fn sample(bank: &Bank, template: usize, band: usize, ph: f64) -> f64 {
        let k = ((ph.rem_euclid(1.0)) * P as f64) as usize % P;
        // recover the sample by inverting the stored transform
        let off = (template * bank.n_band + band) * P;
        let mut buf: Vec<Cx> = bank.fs[off..off + P].to_vec();
        fft(&mut buf, true);
        buf[k].re
    }

    fn make_curve(
        bank: &Bank,
        template: usize,
        period: f64,
        amp: f64,
        phase0: f64,
        n: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<u8>) {
        let (mut t, mut m, mut b) = (vec![], vec![], vec![]);
        for i in 0..n {
            let ti = i as f64 * 0.041;
            let band = i % bank.n_band;
            let ph = (ti / period - phase0).rem_euclid(1.0);
            t.push(ti);
            m.push(17.0 + band as f64 + amp * sample(bank, template, band, ph));
            b.push(band as u8);
        }
        (t, m, b)
    }

    fn trials(true_period: f64, n: usize) -> Vec<f64> {
        let (lo, hi) = (true_period * 0.6, true_period * 1.4);
        let mut v: Vec<f64> = (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
            .collect();
        v.push(true_period);
        v
    }

    #[test]
    fn fft_round_trips() {
        let mut a: Vec<Cx> = (0..16)
            .map(|i| Cx {
                re: (i as f64).sin(),
                im: (i as f64 * 0.3).cos(),
            })
            .collect();
        let orig = a.clone();
        fft(&mut a, false);
        fft(&mut a, true);
        for (x, y) in a.iter().zip(orig.iter()) {
            assert!((x.re - y.re).abs() < 1e-12, "{x:?} vs {y:?}");
            assert!((x.im - y.im).abs() < 1e-12, "{x:?} vs {y:?}");
        }
    }

    #[test]
    fn recovers_known_period() {
        let bank = cosine_bank(&[1.0, 1.0, 1.0]);
        let period = 1.7;
        let (t, m, b) = make_curve(&bank, 0, period, 0.3, 0.0, 600);
        let fit = fit_one(&t, &m, &b, &trials(period, 60), &bank, 12);
        assert_eq!(fit.period, period, "picked {}", fit.period);
        assert!(fit.fvu < 0.05, "fvu {}", fit.fvu);
        assert!(fit.amp > 0.0);
    }

    #[test]
    fn picks_the_sharp_template_when_the_curve_is_sharp() {
        // the point of the whole module: a narrow dip stays narrow
        let bank = cosine_and_eclipse_bank(3);
        let period = 0.83;
        let (t, m, b) = make_curve(&bank, 1, period, 0.4, 0.0, 900);
        let fit = fit_one(&t, &m, &b, &trials(period, 60), &bank, 12);
        assert_eq!(fit.template, 1, "chose template {}", fit.template);
        assert_eq!(fit.period, period);
    }

    #[test]
    fn amplitude_is_shared_across_bands() {
        let bank = cosine_bank(&[1.0, 0.5, 0.25]);
        let period = 1.1;
        let (t, m, b) = make_curve(&bank, 0, period, 0.4, 0.0, 600);
        let per = trials(period, 60);
        let matching = fit_one(&t, &m, &b, &per, &bank, 12).fvu;

        let mut m2 = m.clone();
        for i in 0..m2.len() {
            if b[i] == 2 {
                m2[i] = 19.0 + 4.0 * (m2[i] - 19.0);
            }
        }
        let breaking = fit_one(&t, &m2, &b, &per, &bank, 12).fvu;
        assert!(matching < breaking, "{matching} should beat {breaking}");
    }

    #[test]
    fn phase_shift_is_recovered() {
        let bank = cosine_bank(&[1.0, 1.0, 1.0]);
        let period = 2.3;
        let (t, m, b) = make_curve(&bank, 0, period, 0.3, 0.25, 600);
        let fit = fit_one(&t, &m, &b, &trials(period, 60), &bank, 12);
        assert_eq!(fit.period, period);
        let want = P / 4;
        let diff = (fit.shift as i64 - want as i64).abs();
        assert!(diff <= 2, "shift {} should be near {}", fit.shift, want);
    }

    #[test]
    fn batched_matches_single() {
        let bank = cosine_and_eclipse_bank(3);
        let (mut ts, mut ms, mut bs, mut ps) = (vec![], vec![], vec![], vec![]);
        for (i, p) in [0.9_f64, 1.4, 2.1].iter().enumerate() {
            let (t, m, b) = make_curve(&bank, i % 2, *p, 0.3, 0.0, 400);
            ts.push(t);
            ms.push(m);
            bs.push(b);
            ps.push(trials(*p, 30));
        }
        let batched = fit_batched(&ts, &ms, &bs, &ps, &bank, 12);
        for i in 0..3 {
            let one = fit_one(&ts[i], &ms[i], &bs[i], &ps[i], &bank, 12);
            assert_eq!(batched[i].fvu, one.fvu);
            assert_eq!(batched[i].period, one.period);
            assert_eq!(batched[i].template, one.template);
            assert_eq!(batched[i].shift, one.shift);
        }
    }

    #[test]
    fn too_few_points_gives_no_fit() {
        let bank = cosine_bank(&[1.0, 1.0, 1.0]);
        let (t, m, b) = make_curve(&bank, 0, 1.3, 0.3, 0.0, 9);
        let fit = fit_one(&t, &m, &b, &trials(1.3, 10), &bank, 12);
        assert!(fit.fvu.is_nan(), "fvu {}", fit.fvu);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn bank_rejects_a_grid_that_is_not_a_power_of_two() {
        Bank::new(&vec![0.0; 100], 1, 1, 100);
    }
}
