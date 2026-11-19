/// Ellipsoidal variable fit (BEER model), CPU.
///
/// Model at trial orbital period P, for band b, in flux relative to the band's
/// weighted mean magnitude (Faigler & Mazeh 2011):
///
///     f_b(t) = F_b [1 - Ae_b cos 2φ - Ar_b cos φ + Ab_b sin φ],   φ = 2π (t - T0) / P
///
/// Ae is the ellipsoidal amplitude, Ar the reflection amplitude together with
/// the (R/a)^4 ellipsoidal term, and Ab the Doppler beaming amplitude. T0, the
/// conjunction with the companion in front of the visible star, is shared by
/// all bands. The amplitudes are free per band, so their ratios across bands
/// are an output of the fit.
///
/// At fixed P and T0 the model is linear in (F, F Ae, F Ar, F Ab), and its
/// basis is a rotation of {1, cos ωt, sin ωt, cos 2ωt, sin 2ωt}. Each band's
/// 5x5 weighted Gram matrix is built once per period, after which every trial
/// T0 is a 4x4 solve whose cost does not depend on the number of points.
///
/// Shifting T0 by P/4 flips the sign of cos 2φ and rotates the first harmonic
/// within its own span, so the fitted subspace repeats every quarter cycle and
/// T0 is scanned over [0, P/4). Of the four equivalent values, the one reported
/// has Ae > 0 and Ab > 0, pooled over bands by inverse variance. With no
/// detected beaming, T0 and T0 + P/2 fit equally well and the choice between
/// them is set by noise.
///
/// The comparison model is a sinusoid at P/2 with a free phase per band,
/// {1, cos 2ωt, sin 2ωt}. delta_bic = BIC(sinusoid at P/2) - BIC(BEER) is
/// positive when the odd harmonics (unequal minima and maxima) are detected; a
/// pure cos 2φ light curve gives a negative value, since it is equally well
/// described at half the period.
use std::f64::consts::PI;

use rayon::prelude::*;

use crate::fourier::cholesky_solve;

/// Scalar columns ahead of the per-band block in a result row.
pub const N_SCALAR: usize = 6;
/// Columns per band: Ae, Ae_err, Ab, Ab_err, Ar, Ar_err.
pub const N_PER_BAND: usize = 6;

/// 0.4 ln 10, the derivative of flux with respect to magnitude.
const DFLUX_DMAG: f64 = 0.921_034_037_197_618_3;

/// Each band fits four coefficients, so it needs at least five points.
const MIN_BAND_POINTS: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct BandFit {
    pub a_ellip: f64,
    pub a_ellip_err: f64,
    pub a_beam: f64,
    pub a_beam_err: f64,
    pub a_refl: f64,
    pub a_refl_err: f64,
}

impl BandFit {
    const NAN: BandFit = BandFit {
        a_ellip: f64::NAN,
        a_ellip_err: f64::NAN,
        a_beam: f64::NAN,
        a_beam_err: f64::NAN,
        a_refl: f64::NAN,
        a_refl_err: f64::NAN,
    };
}

/// The best fit for one light curve.
#[derive(Debug, Clone)]
pub struct Fit {
    /// Orbital period, in days.
    pub period: f64,
    /// Conjunction with the companion in front, in the time system of the input.
    pub t0: f64,
    pub chi2: f64,
    /// BIC(sinusoid at P/2) - BIC(BEER at P).
    pub delta_bic: f64,
    /// BIC(constant) - BIC(BEER at P).
    pub delta_bic_flat: f64,
    pub n_points: usize,
    /// One entry per band index; NaN for bands that were not fitted.
    pub bands: Vec<BandFit>,
}

impl Fit {
    fn nan(n_band: usize) -> Fit {
        Fit {
            period: f64::NAN,
            t0: f64::NAN,
            chi2: f64::NAN,
            delta_bic: f64::NAN,
            delta_bic_flat: f64::NAN,
            n_points: 0,
            bands: vec![BandFit::NAN; n_band],
        }
    }

    /// Flatten into `N_SCALAR + N_PER_BAND * n_band` values.
    pub fn write_row(&self, out: &mut [f64]) {
        out[0] = self.period;
        out[1] = self.t0;
        out[2] = self.chi2;
        out[3] = self.delta_bic;
        out[4] = self.delta_bic_flat;
        out[5] = self.n_points as f64;
        for (b, f) in self.bands.iter().enumerate() {
            let o = N_SCALAR + N_PER_BAND * b;
            out[o] = f.a_ellip;
            out[o + 1] = f.a_ellip_err;
            out[o + 2] = f.a_beam;
            out[o + 3] = f.a_beam_err;
            out[o + 4] = f.a_refl;
            out[o + 5] = f.a_refl_err;
        }
    }
}

/// One band as flux minus one, with inverse-variance weights.
struct Band {
    dt: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
}

fn to_flux(t: &[f64], m: &[f64], e: &[f64], t_ref: f64) -> Band {
    let (mut sw, mut swm) = (0.0, 0.0);
    for i in 0..m.len() {
        let w = 1.0 / (e[i] * e[i]);
        sw += w;
        swm += w * m[i];
    }
    let m_ref = swm / sw;

    let n = t.len();
    let mut band = Band {
        dt: Vec::with_capacity(n),
        y: Vec::with_capacity(n),
        w: Vec::with_capacity(n),
    };
    for i in 0..n {
        let f = 10f64.powf(-0.4 * (m[i] - m_ref));
        let sf = DFLUX_DMAG * f * e[i];
        band.dt.push(t[i] - t_ref);
        band.y.push(f - 1.0);
        band.w.push(1.0 / (sf * sf));
    }
    band
}

/// Weighted sums of one band against {1, cos ωt, sin ωt, cos 2ωt, sin 2ωt}.
struct Sums {
    g: [[f64; 5]; 5],
    h: [f64; 5],
    yy: f64,
}

impl Sums {
    fn new(d: &Band, omega: f64) -> Sums {
        let mut g = [[0.0f64; 5]; 5];
        let mut h = [0.0f64; 5];
        let mut yy = 0.0;
        for i in 0..d.y.len() {
            let (s1, c1) = (omega * d.dt[i]).sin_cos();
            let u = [1.0, c1, s1, c1 * c1 - s1 * s1, 2.0 * s1 * c1];
            let (w, y) = (d.w[i], d.y[i]);
            for a in 0..5 {
                h[a] += w * y * u[a];
                for b in 0..=a {
                    g[a][b] += w * u[a] * u[b];
                }
            }
            yy += w * y * y;
        }
        for a in 0..5 {
            for b in (a + 1)..5 {
                g[a][b] = g[b][a];
            }
        }
        Sums { g, h, yy }
    }

    /// Weighted least squares on the basis functions given as rows of `m`,
    /// each a combination of the five functions above. Writes the coefficients
    /// to `x` and, when asked, the covariance (row-major, k x k) to `cov`.
    /// Returns chi2, or None if the normal matrix is singular.
    fn solve(&self, m: &[[f64; 5]], x: &mut [f64], cov: Option<&mut [f64]>) -> Option<f64> {
        let k = m.len();
        let mut a = [0.0f64; 16];
        let mut r = [0.0f64; 4];
        for i in 0..k {
            let mut gm = [0.0f64; 5];
            for p in 0..5 {
                for q in 0..5 {
                    gm[p] += self.g[p][q] * m[i][q];
                }
            }
            for j in 0..k {
                a[i * k + j] = (0..5).map(|p| m[j][p] * gm[p]).sum();
            }
            r[i] = (0..5).map(|p| m[i][p] * self.h[p]).sum();
        }

        let mut work = a;
        let mut sol = r;
        if !cholesky_solve(&mut work[..k * k], &mut sol[..k], k) {
            return None;
        }
        x[..k].copy_from_slice(&sol[..k]);

        if let Some(cov) = cov {
            for j in 0..k {
                let mut work = a;
                let mut e = [0.0f64; 4];
                e[j] = 1.0;
                if !cholesky_solve(&mut work[..k * k], &mut e[..k], k) {
                    return None;
                }
                for i in 0..k {
                    cov[i * k + j] = e[i];
                }
            }
        }

        let chi2 = self.yy - (0..k).map(|i| sol[i] * r[i]).sum::<f64>();
        Some(chi2.max(0.0))
    }
}

/// Rows {1, cos 2φ, cos φ, sin φ} for φ = ωt - theta.
fn beer_basis(theta: f64) -> [[f64; 5]; 4] {
    let (s, c) = theta.sin_cos();
    let (s2, c2) = (2.0 * theta).sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c2, s2],
        [0.0, c, s, 0.0, 0.0],
        [0.0, -s, c, 0.0, 0.0],
    ]
}

const HALF_BASIS: [[f64; 5]; 3] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
];

const FLAT_BASIS: [[f64; 5]; 1] = [[1.0, 0.0, 0.0, 0.0, 0.0]];

fn pooled_chi2(sums: &[Sums], m: &[[f64; 5]]) -> Option<f64> {
    let mut x = [0.0f64; 4];
    sums.iter().map(|s| s.solve(m, &mut x, None)).sum()
}

/// Best theta = ω T0 in [0, π/2) and its chi2. The pooled chi2 repeats every
/// π/2, so the grid wraps and the minimum is refined by a parabola through its
/// neighbours.
fn scan_t0(sums: &[Sums], n_t0: usize) -> Option<(f64, f64)> {
    let n = n_t0.max(4);
    let step = 0.5 * PI / n as f64;
    let chi: Vec<f64> = (0..n)
        .map(|j| pooled_chi2(sums, &beer_basis(j as f64 * step)).unwrap_or(f64::INFINITY))
        .collect();
    let (j, &c0) = chi.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))?;
    if !c0.is_finite() {
        return None;
    }

    let mut best = (j as f64 * step, c0);
    let (cm, cp) = (chi[(j + n - 1) % n], chi[(j + 1) % n]);
    let denom = cm - 2.0 * c0 + cp;
    if denom > 0.0 && denom.is_finite() {
        let delta = (0.5 * (cm - cp) / denom).clamp(-1.0, 1.0);
        let theta = (j as f64 + delta) * step;
        if let Some(c) = pooled_chi2(sums, &beer_basis(theta)) {
            if c < c0 {
                best = (theta, c);
            }
        }
    }
    Some(best)
}

/// value = sign * a / F, with its standard error from the 4x4 covariance of
/// (F - 1, a2, a1c, a1s); `i` is the index of `a`.
fn amplitude(x: &[f64; 4], cov: &[f64; 16], i: usize, sign: f64) -> (f64, f64) {
    let f = 1.0 + x[0];
    let a = x[i];
    let var = cov[i * 4 + i] / (f * f) + a * a * cov[0] / (f * f * f * f)
        - 2.0 * a * cov[i * 4] / (f * f * f);
    (sign * a / f, var.max(0.0).sqrt())
}

/// Inverse-variance weighted mean.
fn pooled(values: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut sw, mut swv) = (0.0, 0.0);
    for (v, e) in values {
        if e > 0.0 && e.is_finite() && v.is_finite() {
            sw += 1.0 / (e * e);
            swv += v / (e * e);
        }
    }
    swv / sw
}

/// Per-band amplitudes at one theta, or None if a band is singular.
fn band_fits(sums: &[Sums], theta: f64) -> Option<Vec<BandFit>> {
    let m = beer_basis(theta);
    sums.iter()
        .map(|s| {
            let mut x = [0.0f64; 4];
            let mut cov = [0.0f64; 16];
            s.solve(&m, &mut x, Some(&mut cov))?;
            let (a_ellip, a_ellip_err) = amplitude(&x, &cov, 1, -1.0);
            let (a_refl, a_refl_err) = amplitude(&x, &cov, 2, -1.0);
            let (a_beam, a_beam_err) = amplitude(&x, &cov, 3, 1.0);
            Some(BandFit {
                a_ellip,
                a_ellip_err,
                a_beam,
                a_beam_err,
                a_refl,
                a_refl_err,
            })
        })
        .collect()
}

/// Fit one light curve at each of its trial periods and return the best.
///
/// With `double_periods`, each trial period is taken as a periodogram peak at
/// P_orb / 2 and the fit runs at twice it.
#[allow(clippy::too_many_arguments)]
pub fn fit_one(
    times: &[f64],
    mags: &[f64],
    errs: &[f64],
    bands: &[u8],
    periods: &[f64],
    n_band: usize,
    n_t0: usize,
    double_periods: bool,
    min_points: usize,
) -> Fit {
    let mut bt: Vec<Vec<f64>> = vec![Vec::new(); n_band];
    let mut bm: Vec<Vec<f64>> = vec![Vec::new(); n_band];
    let mut be: Vec<Vec<f64>> = vec![Vec::new(); n_band];
    for i in 0..times.len() {
        let b = bands[i] as usize;
        let (t, m, e) = (times[i], mags[i], errs[i]);
        if b < n_band && t.is_finite() && m.is_finite() && e.is_finite() && e > 0.0 {
            bt[b].push(t);
            bm[b].push(m);
            be[b].push(e);
        }
    }

    let need = min_points.max(MIN_BAND_POINTS);
    let used: Vec<usize> = (0..n_band).filter(|&b| bt[b].len() >= need).collect();
    if used.is_empty() {
        return Fit::nan(n_band);
    }

    let t_ref = used
        .iter()
        .flat_map(|&b| bt[b].iter().copied())
        .fold(f64::INFINITY, f64::min);
    let data: Vec<Band> = used
        .iter()
        .map(|&b| to_flux(&bt[b], &bm[b], &be[b], t_ref))
        .collect();
    let n_points: usize = data.iter().map(|d| d.y.len()).sum();

    let mut best: Option<(f64, f64, f64, Vec<Sums>)> = None;
    for &p_in in periods {
        let p = if double_periods { 2.0 * p_in } else { p_in };
        if !(p.is_finite() && p > 0.0) {
            continue;
        }
        let sums: Vec<Sums> = data.iter().map(|d| Sums::new(d, 2.0 * PI / p)).collect();
        if let Some((theta, chi2)) = scan_t0(&sums, n_t0) {
            let better = match &best {
                None => true,
                Some(b) => chi2 < b.2,
            };
            if better {
                best = Some((p, theta, chi2, sums));
            }
        }
    }
    let Some((period, theta, chi2, sums)) = best else {
        return Fit::nan(n_band);
    };

    // Of the four equivalent T0, keep the one with positive pooled Ae and the
    // larger pooled Ab.
    let mut pick: Option<(f64, Vec<BandFit>, bool, f64)> = None;
    for q in 0..4 {
        let th = theta + q as f64 * 0.5 * PI;
        let Some(fits) = band_fits(&sums, th) else {
            return Fit::nan(n_band);
        };
        let ae = pooled(fits.iter().map(|f| (f.a_ellip, f.a_ellip_err))) > 0.0;
        let ab = pooled(fits.iter().map(|f| (f.a_beam, f.a_beam_err)));
        let better = match &pick {
            None => true,
            Some((_, _, pae, pab)) => (ae && !pae) || (ae == *pae && ab > *pab),
        };
        if better {
            pick = Some((th, fits, ae, ab));
        }
    }
    let (theta, fits, _, _) = pick.expect("four candidates were evaluated");

    let ln_n = (n_points as f64).ln();
    let nb = used.len() as f64;
    let bic_beer = chi2 + (4.0 * nb + 1.0) * ln_n;
    let bic_half = pooled_chi2(&sums, &HALF_BASIS).map_or(f64::NAN, |c| c + 3.0 * nb * ln_n);
    let bic_flat = pooled_chi2(&sums, &FLAT_BASIS).map_or(f64::NAN, |c| c + nb * ln_n);

    let mut out_bands = vec![BandFit::NAN; n_band];
    for (k, &b) in used.iter().enumerate() {
        out_bands[b] = fits[k];
    }

    Fit {
        period,
        t0: t_ref + theta.rem_euclid(2.0 * PI) / (2.0 * PI) * period,
        chi2,
        delta_bic: bic_half - bic_beer,
        delta_bic_flat: bic_flat - bic_beer,
        n_points,
        bands: out_bands,
    }
}

/// Fit a batch of light curves in parallel, one object per thread task.
#[allow(clippy::too_many_arguments)]
pub fn fit_batched(
    times: &[Vec<f64>],
    mags: &[Vec<f64>],
    errs: &[Vec<f64>],
    bands: &[Vec<u8>],
    periods: &[Vec<f64>],
    n_band: usize,
    n_t0: usize,
    double_periods: bool,
    min_points: usize,
) -> Vec<Fit> {
    (0..times.len())
        .into_par_iter()
        .map(|i| {
            fit_one(
                &times[i],
                &mags[i],
                &errs[i],
                &bands[i],
                &periods[i],
                n_band,
                n_t0,
                double_periods,
                min_points,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64* with Box-Muller, so the tests need no extra crates.
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-300);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }

    struct Curve {
        t: Vec<f64>,
        m: Vec<f64>,
        e: Vec<f64>,
        b: Vec<u8>,
    }

    /// BEER light curve; `amps[b]` is (Ae, Ar, Ab) for band b.
    fn make_curve(
        period: f64,
        t0: f64,
        amps: &[(f64, f64, f64)],
        n: usize,
        noise: f64,
        seed: u64,
    ) -> Curve {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut c = Curve {
            t: vec![],
            m: vec![],
            e: vec![],
            b: vec![],
        };
        for _ in 0..n {
            let t = 60000.0 + 1000.0 * rng.uniform();
            let b = ((rng.uniform() * amps.len() as f64) as usize).min(amps.len() - 1);
            let (ae, ar, ab) = amps[b];
            let phi = 2.0 * PI * (t - t0) / period;
            let f = 1.0 - ae * (2.0 * phi).cos() - ar * phi.cos() + ab * phi.sin();
            c.t.push(t);
            c.m.push(15.0 + b as f64 - 2.5 * f.log10() + noise * rng.normal());
            c.e.push(noise);
            c.b.push(b as u8);
        }
        c
    }

    fn run(c: &Curve, periods: &[f64], n_band: usize) -> Fit {
        fit_one(&c.t, &c.m, &c.e, &c.b, periods, n_band, 128, true, 12)
    }

    /// Distance between two times modulo the period, as a fraction of it.
    fn phase_diff(a: f64, b: f64, period: f64) -> f64 {
        let d = ((a - b) / period).rem_euclid(1.0);
        d.min(1.0 - d)
    }

    const AMPS: [(f64, f64, f64); 3] = [
        (0.030, 0.004, 0.002),
        (0.025, 0.003, 0.0015),
        (0.022, 0.002, 0.0012),
    ];

    #[test]
    fn recovers_amplitudes_and_t0() {
        let (period, t0) = (0.37, 60000.123);
        let c = make_curve(period, t0, &AMPS, 3000, 0.001, 1);
        let fit = run(&c, &[period / 2.0], 3);

        assert!((fit.period - period).abs() < 1e-12);
        assert!(phase_diff(fit.t0, t0, period) < 2e-3, "t0 {}", fit.t0);
        for (b, &(ae, ar, ab)) in AMPS.iter().enumerate() {
            let f = fit.bands[b];
            assert!(
                (f.a_ellip - ae).abs() < 5.0 * f.a_ellip_err,
                "band {b}: Ae {} vs {ae}",
                f.a_ellip
            );
            assert!(
                (f.a_refl - ar).abs() < 5.0 * f.a_refl_err,
                "band {b}: Ar {} vs {ar}",
                f.a_refl
            );
            assert!(
                (f.a_beam - ab).abs() < 5.0 * f.a_beam_err,
                "band {b}: Ab {} vs {ab}",
                f.a_beam
            );
        }
        assert!(fit.delta_bic > 0.0);
        assert!(fit.delta_bic_flat > fit.delta_bic);
    }

    #[test]
    fn beaming_sets_the_half_orbit_branch() {
        let period = 0.37;
        for t0 in [60000.05, 60000.05 + period / 2.0, 60000.05 + period / 4.0] {
            let c = make_curve(period, t0, &AMPS, 3000, 0.001, 2);
            let fit = run(&c, &[period / 2.0], 3);
            assert!(
                phase_diff(fit.t0, t0, period) < 2e-3,
                "t0 {} vs {t0}",
                fit.t0
            );
            assert!(fit.bands[0].a_ellip > 0.0 && fit.bands[0].a_beam > 0.0);
        }
    }

    #[test]
    fn pure_ellipsoidal_prefers_half_period() {
        let amps = [(0.03, 0.0, 0.0), (0.03, 0.0, 0.0)];
        let c = make_curve(0.5, 60000.2, &amps, 2000, 0.001, 3);
        let fit = run(&c, &[0.25], 2);
        assert!(fit.delta_bic < 0.0, "delta_bic {}", fit.delta_bic);
    }

    #[test]
    fn picks_orbital_period_among_candidates() {
        let period = 0.37;
        let c = make_curve(period, 60000.1, &AMPS, 3000, 0.001, 4);
        let fit = run(&c, &[0.21, period / 2.0, 0.29, period], 3);
        assert!((fit.period - period).abs() < 1e-12, "period {}", fit.period);
    }

    #[test]
    fn unused_band_is_nan() {
        let c = make_curve(0.37, 60000.1, &AMPS[..2], 2000, 0.001, 5);
        let fit = run(&c, &[0.185], 3);
        assert!(fit.bands[0].a_ellip.is_finite());
        assert!(fit.bands[1].a_ellip.is_finite());
        assert!(fit.bands[2].a_ellip.is_nan());
    }

    #[test]
    fn no_usable_band_is_nan() {
        let c = make_curve(0.37, 60000.1, &AMPS, 6, 0.001, 6);
        let fit = run(&c, &[0.185], 3);
        assert!(fit.period.is_nan());
        assert_eq!(fit.n_points, 0);
    }
}
